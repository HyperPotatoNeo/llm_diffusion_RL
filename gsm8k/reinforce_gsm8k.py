import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from generate import batch_generate
import deepspeed
from deepspeed.utils import RepeatingLoader
import numpy as np
from datasets import load_dataset
from typing import List, Dict, Any, Tuple
import logging
import argparse
import json
from tqdm import tqdm
import random
import math
import re
import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MASK_ID = 126336

def setup_args():
    parser = argparse.ArgumentParser(description="DeepSpeed REINFORCE Trainer for LLM on GSM8K")
    parser.add_argument("--model_name_or_path", type=str, default='GSAI-ML/LLaDA-8B-Instruct', help="Path to pretrained model")
    parser.add_argument("--output_dir", type=str, default="./gsm8k_output", help="Output directory")
    parser.add_argument("--data_split", type=str, default="train", help="GSM8K data split to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--responses_per_gpu", type=int, default=4, help="Max responses to process per GPU")
    parser.add_argument("--k_samples", type=int, default=4, help="Number of samples per query for baseline")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.6, help="Training policy temperature")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every X steps")
    parser.add_argument("--eval_freq", type=int, default=30, help="Evaluation frequency in steps")
    parser.add_argument("--eval_size", type=int, default=100, help="Number of examples to evaluate on")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--drop_prob", type=float, default=0.5, help="probability of dropping timestep during policy update")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[2, 3],
                        help="ZeRO optimization stage to use (2 or 3)")
    parser.add_argument("--ds_config", type=str, default='gsm8k/reinforce_ds_config.json', help="DeepSpeed config file")
    return parser.parse_args()

def setup_deepspeed():
    """Initialize the DeepSpeed environment."""
    deepspeed.init_distributed()
    args = setup_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Create output directory on rank 0
    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def load_gsm8k_dataset(args):
    """Load the GSM8K dataset."""
    dataset = load_dataset('openai/gsm8k', 'main')
    dataset = dataset[args.data_split]
    return dataset

def prepare_batch_for_gpu(batch: Dict[str, List[Any]], tokenizer):
    questions = batch["question"]  # This is a list of strings
    formatted_questions = []
    
    # Process each question separately
    for question in questions:
        m = [{"role": "user", "content": question}]
        formatted = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        formatted_questions.append(formatted)
    encodings = tokenizer(formatted_questions)
    
    max_length = max(len(item) for item in encodings['input_ids'])
    # Right-pad all inputs to the same length
    for i in range(len(encodings['input_ids'])):
        padding_length = max_length - len(encodings['input_ids'][i])
        # Right pad with mask_id
        encodings['input_ids'][i] = encodings['input_ids'][i] + [MASK_ID] * padding_length
    encodings['input_ids'] = torch.tensor(encodings['input_ids']).to(torch.device(f"cuda:{dist.get_rank()}"))
    return encodings#.to(torch.device(f"cuda:{dist.get_rank()}"))

def create_reasoning_prompts(data):
    """
    Create prompts for a list of questions and answers.
    
    Args:
        data (list of tuples): Each tuple should be of the form 
            (question, llada_response, ground_truth).
    
    Returns:
        list of str: A list where each element is a prompt for the Qwen model.
    """
    prompts = []
    for question, llada_response, ground_truth in data:
        prompt = f"""You are a helpful assistant that evaluates math solutions. 
I'll provide you with a math problem, a proposed solution, and the correct final answer.

Your task is to:
1. Carefully check if the proposed solution arrives at the correct final answer
2. Reason step by step to verify if the solution is correct
3. Determine if the final numerical answer in the solution matches the ground truth answer
4. Conclude with a YES or NO answer only to indicate whether the solution is correct

Math Problem:
{question}

Proposed Solution:
{llada_response}

Ground Truth Answer: {ground_truth}

Please think carefully step by step and finally just answer YES or NO to whether the solution arrived at the correct final answer.
"""
        prompts.append(prompt)
    return prompts


class ReinforceTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{dist.get_rank()}")
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Load model and tokenizer. Using AutoModelForCausalLM for generation and loss computation.
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        
        # Configure DeepSpeed with either a provided config file or a default one
        ds_config = self._prepare_deepspeed_config()
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config
        )
        self.model_engine.train()
        
        # Load reward verification model
        self.reward_model_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True, padding_side="left")
        self.reward_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.reward_model.to(self.model_engine.device)
        self.reward_model.eval()
        
        # Dataset
        self.dataset = load_gsm8k_dataset(args)
        
        # Training tracking
        self.global_step = 0
        self.epoch = 0
        
        # Calculate effective batch size and steps
        self.effective_batch_size = args.batch_size * self.world_size
        dataset_size = len(self.dataset)
        self.steps_per_epoch = math.ceil(dataset_size / self.effective_batch_size)
        self.total_steps = self.steps_per_epoch * args.num_train_epochs
        
        logger.info(f"Rank {self.global_rank} initialized. World size: {self.world_size}")
        logger.info(f"Dataset size: {dataset_size}, Steps per epoch: {self.steps_per_epoch}")
        logger.info(f"Effective batch size: {self.effective_batch_size}, Total steps: {self.total_steps}")
        
    def reward_function(self, reference_questions: List[str], generated_responses: List[str], reference_answers: List[str]) -> List[float]:
        """
        LLM eval reward function
        """
        ground_truth = [re.search(r'####\s*([\d\.\-\+]+)', answer).group(1).strip() for answer in reference_answers]
        eval_prompts = create_reasoning_prompts(list(zip(reference_questions, generated_responses, ground_truth)))
        formatted_questions = []
        for prompt in eval_prompts:
            m = [{"role": "user", "content": prompt}]
            formatted = self.reward_model_tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            formatted_questions.append(formatted)      
        eval_prompts_encoded = self.reward_model_tokenizer(formatted_questions, return_tensors="pt", padding=True, truncation=True).to(torch.device(f"cuda:{dist.get_rank()}"))
        with torch.no_grad():
            outputs = self.reward_model.generate(
                eval_prompts_encoded.input_ids
            )
        eval_responses = self.reward_model_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        eval_responses = [resp.lower().split("\nassistant\n", 1)[-1].strip() if "\nassistant\n" in resp.lower() else resp.strip() for resp in eval_responses]
        #rewards = torch.tensor([1.0 if "YES" in response.strip().upper() else 0.0 for response in eval_responses]).to(torch.device(f"cuda:{dist.get_rank()}"))
        rewards = [1.0 if "YES" in response.strip().upper() else 0.0 for response in eval_responses]

        return rewards
    
    def _prepare_deepspeed_config(self):
        """Prepare DeepSpeed configuration, overriding with the stage specified by --zero_stage if applicable."""
        if self.args.ds_config and os.path.exists(self.args.ds_config):
            with open(self.args.ds_config, "r") as f:
                ds_config = json.load(f)
            # Override the ZeRO stage with the command-line argument
            ds_config["zero_optimization"]["stage"] = self.args.zero_stage
        else:
            # Build a default DeepSpeed configuration
            ds_config = {
                "train_batch_size": self.args.batch_size * self.world_size,
                "gradient_accumulation_steps": self.args.grad_accum_steps,
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": self.args.learning_rate,
                        "weight_decay": self.args.weight_decay,
                        "betas": [0.9, 0.999],
                        "eps": 1e-8
                    }
                },
                "bf16": {
                    "enabled": True
                },
                "zero_optimization": {
                    "stage": self.args.zero_stage
                }
            }
            # Adjust configuration based on ZeRO stage
            if self.args.zero_stage == 2:
                # For Stage 2, enable CPU offloading with a boolean value.
                pass
                #ds_config["zero_optimization"]["cpu_offload"] = False
            elif self.args.zero_stage == 3:
                # For Stage 3, offload both parameters and optimizer states to CPU,
                # along with additional stage-3-specific settings.
                ds_config["zero_optimization"].update({
                    "offload_param": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": 50000000,
                    "allgather_bucket_size": 50000000
                })
        return ds_config

    
    def _get_dataloader(self):
        """Create a distributed dataloader with DistributedSampler."""
        sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=True)
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            drop_last=True  # Ensures equal-sized batches across GPUs
        )
        return dataloader
    
    def generate_responses(self, input_ids):
        """
        Generate K responses for each query in the batch.
        This version processes queries in chunks and extracts only the generated continuation.
        """
        batch_size = input_ids.size(0)
        all_responses = []
        
        total_responses_needed = batch_size * self.args.k_samples
        max_responses_per_iter = min(self.args.responses_per_gpu, total_responses_needed)
        queries_per_iter = max_responses_per_iter // self.args.k_samples
        num_iterations = math.ceil(total_responses_needed / max_responses_per_iter)
        
        for i in range(num_iterations):
            start_idx = i * queries_per_iter
            end_idx = min(start_idx + queries_per_iter, batch_size)
            if start_idx >= end_idx:
                break
            
            # Select queries for this chunk
            selected_input_ids = input_ids[start_idx:end_idx]
            
            # Repeat each query k_samples times
            repeated_input_ids = selected_input_ids.repeat_interleave(self.args.k_samples, dim=0)
            
            with torch.no_grad():
                outputs, traj = batch_generate(self.model_engine, prompt=repeated_input_ids, gen_length=256, block_length=32, steps=32, remasking='random', temperature=self.args.temperature)
            # For each query in the chunk, extract responses
            for j in range(selected_input_ids.size(0)):
                prompt_ids = selected_input_ids[j]
                prompt_len = (prompt_ids != MASK_ID).sum().item()
                for k in range(self.args.k_samples):
                    overall_idx = j * self.args.k_samples + k
                    generated_ids = outputs[overall_idx]
                    response_ids = generated_ids[prompt_len:]
                    traj_idx = traj[overall_idx] # size T * seq_len
                    
                    decoded_response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    all_responses.append({
                        "query_idx": start_idx + j,
                        "sample_idx": k,
                        "response": decoded_response,
                        "response_ids": response_ids.tolist(),
                        "prompt_ids": prompt_ids.tolist(),
                        "traj_idx": traj_idx.tolist()
                    })
        return all_responses
    
    def compute_policy_loss(self, responses, advantages):
        """
        Compute the REINFORCE policy gradient loss by computing the log probability
        of tokens that are unmasked at each generation timestep.

        For each generation step, we:
        1. Gather all samples (from a group of k_samples) that have a newly unmasked token.
        2. Compute the loss in batch by constructing a full target tensor from the per-sample
            target tokens and corresponding boolean masks.
        3. Immediately call backward on the batched loss so that the computation graph for that
            step is freed, saving memory.
        4. Update a cached input for each sample with the new token so that future timesteps
            see the updated sequence.

        Temporary tensors are deleted as soon as possible to free memory.
        """
        total_loss_value = 0.0  # For logging purposes.
        processed_count = 0

        # Process responses in groups of k_samples.
        for i in range(0, len(responses), self.args.k_samples):
            # Get current group responses and corresponding advantages.
            batch_responses = responses[i : i + self.args.k_samples]
            batch_advantages = advantages[i : i + self.args.k_samples]
            # Normalize advantages (results in a list of floats).
            batch_advantages = (batch_advantages - np.mean(batch_advantages)) / (np.std(batch_advantages) + 1e-4)
            #if all(adv == 0 for adv in batch_advantages):
            #    continue

            # Determine the number of generation timesteps from the first response's trajectory.
            first_traj = torch.tensor(batch_responses[0]["traj_idx"], device=self.device)
            num_steps = first_traj.size(0)
            del first_traj  # Free temporary tensor

            # Preinitialize cached input for each sample based on the prompt.
            cached_inputs = []
            for response in batch_responses:
                traj_example = torch.tensor(response["traj_idx"], device=self.device)
                seq_length = traj_example.shape[1]
                del traj_example
                prompt_ids = torch.tensor(response["prompt_ids"], device=self.device)
                init_input = torch.full((seq_length,), MASK_ID, device=self.device)
                init_input[:len(prompt_ids)] = prompt_ids
                cached_inputs.append(init_input.clone())
                del prompt_ids, init_input

            # Process each generation timestep.
            for step in range(num_steps):
                print('Iter: ', i ,' STEP: ', step)
                batch_inputs = []
                batch_targets = []    # Each is a 1D tensor holding the target token(s) unmasked this step.
                batch_masks = []      # Boolean mask (shape: [seq_length]) for positions updated.
                batch_adv_list = []   # Advantage for each sample in the current batch.

                # For each sample in the group, collect data if a token is unmasked at this step.
                for j, (response, advantage) in enumerate(zip(batch_responses, batch_advantages)):
                    response_traj = torch.tensor(response["traj_idx"], device=self.device)
                    input_seq = cached_inputs[j].clone()

                    # Determine positions that are being unmasked at the current timestep.
                    current_mask = (response_traj[step] != MASK_ID)
                    if not current_mask.any():
                        del response_traj, input_seq, current_mask
                        continue  # Skip if no token is unmasked at this step.

                    target_tokens = response_traj[step][current_mask]

                    batch_inputs.append(input_seq)
                    batch_targets.append(target_tokens)
                    batch_masks.append(current_mask)
                    batch_adv_list.append(advantage)

                    # Update the cached input for this sample.
                    updated_input = input_seq.clone()
                    updated_input[current_mask] = response_traj[step][current_mask]
                    cached_inputs[j] = updated_input

                    del response_traj, input_seq, current_mask, updated_input

                if len(batch_inputs) == 0:
                    continue
                
                # Continue with probability drop_prob (for training efficiency)
                if random.random() < self.args.drop_prob:
                    continue

                # Stack the collected input sequences to form a batch tensor.
                batch_inputs_tensor = torch.stack(batch_inputs)  # [B, seq_length]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model_engine(input_ids=batch_inputs_tensor)
                    logits = outputs.logits  # [B, seq_length, vocab_size]
                del batch_inputs_tensor, outputs

                B, seq_length, vocab_size = logits.shape

                # Stack boolean masks into a tensor of shape [B, seq_length].
                stacked_masks = torch.stack(batch_masks, dim=0).to(self.device)  # [B, seq_length]

                # Build a full target tensor of shape [B, seq_length] to hold the target tokens.
                full_targets = torch.zeros((B, seq_length), dtype=torch.long, device=self.device)
                for idx in range(B):
                    indices = batch_masks[idx].nonzero(as_tuple=False).squeeze(-1)
                    if indices.numel() > 0:
                        full_targets[idx].scatter_(0, indices, batch_targets[idx])
                del batch_masks, batch_targets

                # Compute log probabilities over the vocabulary.
                log_probs_all = F.log_softmax(logits/self.args.temperature, dim=-1)  # [B, seq_length, vocab_size]
                del logits
                # Gather log probabilities corresponding to target tokens.
                gathered = torch.gather(log_probs_all, dim=-1, index=full_targets.unsqueeze(-1)).squeeze(-1)  # [B, seq_length]
                del full_targets, log_probs_all
                # Zero out positions that were not unmasked.
                masked_log_probs = gathered * stacked_masks.float()  # [B, seq_length]
                del gathered, stacked_masks
                # Sum log probabilities over sequence for each sample.
                per_sample_loss = -masked_log_probs.sum(dim=1)  # [B]
                del masked_log_probs
                # Convert the list of advantages to a tensor.
                batch_adv_tensor = torch.tensor(batch_adv_list, device=self.device, dtype=per_sample_loss.dtype)
                weighted_losses = per_sample_loss * batch_adv_tensor  # [B]
                total_weighted_loss = weighted_losses.sum()  # Scalar

                # Immediately backpropagate the batched loss.
                self.model_engine.backward(total_weighted_loss)

                total_loss_value += total_weighted_loss.item()
                processed_count += B

                # Clean up temporary tensors for this timestep.
                del batch_adv_tensor, weighted_losses, total_weighted_loss, per_sample_loss

            # End of timesteps: clean up cached tensors.
            del cached_inputs, batch_responses, batch_advantages

        if processed_count > 0:
            avg_loss = total_loss_value/len(responses) #/ processed_count
            return avg_loss
        return 0.0

    
    def train_step(self, batch):
        """Perform one training step."""
        self.model_engine.zero_grad()
        inputs = prepare_batch_for_gpu(batch, self.tokenizer)
        responses = self.generate_responses(inputs["input_ids"])
        
        generated_texts = [resp["response"] for resp in responses]
        reference_questions = [batch["question"][resp["query_idx"]] for resp in responses]
        reference_answers = [batch["answer"][resp["query_idx"]] for resp in responses]
        rewards = self.reward_function(reference_questions, generated_texts, reference_answers)
        idx = 0
        for resp in responses:
            resp["reward"] = rewards[idx]
            idx += 1
        
        #advantages = self.sync_advantages_across_gpus(responses, rewards)
        print('COMPUTING LOSS: ')
        loss = self.compute_policy_loss(responses, rewards)

        self.model_engine.step()
        
        return torch.tensor(loss).to(self.device), torch.tensor(rewards).to(self.device).mean()
    
    def train(self):
        """Main training loop with DistributedSampler and periodic evaluation."""
        logger.info("Starting training...")
        if dist.get_rank() == 0:
            wandb.init(project="llada-gsm8k", config=vars(self.args))

        # Add evaluation frequency to args or use default
        eval_freq = getattr(self.args, "eval_freq", 100)

        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            epoch_loss = 0
            steps_this_epoch = 0

            dataloader = self._get_dataloader()
            sampler = dataloader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)

            progress_bar = tqdm(range(self.steps_per_epoch), disable=self.global_rank != 0)

            for step, batch in enumerate(dataloader):
                if step >= self.steps_per_epoch:
                    break

                # Run evaluation at specified frequency
                self.evaluate()
                
                loss, rewards = self.train_step(batch)
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(rewards, op=dist.ReduceOp.AVG)
                epoch_loss += loss
                steps_this_epoch += 1
                self.global_step += 1

                if self.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(f"Epoch {epoch}, Loss: {loss}")
                    
                if dist.get_rank() == 0:
                    wandb.log({
                        "loss": loss.detach().cpu(),
                        "epoch": epoch,
                        "step": step,
                        "avg_reward": rewards.detach().cpu()
                    })

                if self.global_step % self.args.save_steps == 0:
                    pass
                    #self.save_checkpoint()

            avg_epoch_loss = epoch_loss / steps_this_epoch
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Run evaluation at the end of each epoch
            self.evaluate()

    logger.info("Training completed.")
    
    def evaluate(self):
        """
        Evaluate the model on the test set in batches and report success rate (average reward).
        
        Args:
            eval_freq (int): How often to run evaluation (in steps)
        
        Returns:
            float: Average reward (success rate) on the test set
        """
        if self.global_step % self.args.eval_freq != 0:
            return None
        
        logger.info(f"Running evaluation at step {self.global_step}...")
        
        # Switch to evaluation mode
        self.model_engine.eval()
        
        # Load the test dataset
        eval_dataset = load_dataset('openai/gsm8k', 'main')['test']
        
        # Create a distributed sampler for the test set
        eval_sampler = DistributedSampler(
            eval_dataset, 
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=False
        )
        
        # Create a dataloader with the sampler
        eval_batch_size = 8#min(8, self.args.batch_size * 2)  # Can use larger batch size for evaluation
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            sampler=eval_sampler,
            drop_last=False
        )
        
        # Track total and correct predictions
        total_rewards = []
        
        # Process batches
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=self.global_rank != 0):
            # Prepare the batch using the existing function
            inputs = prepare_batch_for_gpu(batch, self.tokenizer)
            
            # Generate responses (one per input)
            with torch.no_grad():
                outputs, _ = batch_generate(
                    self.model_engine, 
                    prompt=inputs["input_ids"], 
                    gen_length=256, 
                    block_length=32, 
                    steps=32, 
                    remasking='random', 
                    temperature=0.0  # Using a higher temperature for generation
                )
            
            # Process each sample in the batch
            for i, (question, answer, output) in enumerate(zip(batch["question"], batch["answer"], outputs)):
                # Determine the prompt length to extract only the generated content
                prompt_ids = inputs["input_ids"][i]
                prompt_len = (prompt_ids != MASK_ID).sum().item()
                
                # Extract the generated response
                response_ids = output[prompt_len:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                # Calculate reward
                reward = self.reward_function([question], [response], [answer])[0]
                total_rewards.append(reward)
        
        # Gather rewards from all GPUs
        all_rewards = [None for _ in range(self.world_size)]
        dist.all_gather_object(all_rewards, total_rewards)
        
        # Calculate the average reward
        if self.global_rank == 0:
            # Flatten the list of rewards
            flat_rewards = [r for sublist in all_rewards for r in sublist if r is not None]
            average_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0
            
            logger.info(f"Evaluation results at step {self.global_step}: Average reward = {average_reward:.4f} (Success rate: {average_reward*100:.2f}%)")
            
            # Log to wandb
            wandb.log({
                "eval_reward": average_reward,
                "eval_success_rate": average_reward * 100,  # as percentage
                "global_step": self.global_step
            })
            
            # Switch back to training mode
            self.model_engine.train()
            return average_reward
        
        # Switch back to training mode
        self.model_engine.train()
        return None
    
    def save_checkpoint(self, suffix=""):
        """Save model checkpoint."""
        if self.global_rank == 0:
            output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
            if suffix:
                output_dir = f"{output_dir}-{suffix}"
            os.makedirs(output_dir, exist_ok=True)
            self.model_engine.save_checkpoint(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                json.dump(vars(self.args), f, indent=2)
            logger.info(f"Saved checkpoint to {output_dir}")

def main():
    args = setup_deepspeed()
    trainer = ReinforceTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
