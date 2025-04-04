from typing import List
import os
import argparse
import logging
import math
import random
import torch
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
import numpy as np
import wandb
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from tqdm import tqdm
import re
from collections import deque
import json

os.environ["HF_HOME"] = "/home/mila/j/jain.vineet/scratch/.cache/huggingface/"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"] = "false"

from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForCausalLM,
    get_scheduler
)
from datasets import load_dataset
from dataloader import GSM8KDataset, collate_fn_factory, BlockGSM8KDataset, block_collate_fn_factory
from generate import batch_generate

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Global mask token ID
MASK_ID = 126336

def setup_args():
    parser = argparse.ArgumentParser(description="Reward-Weighted Regression (RWR) of LLaDA-8B on GSM8K using DeepSpeed")
    
    # Model and dataset arguments
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Path to pretrained model")
    parser.add_argument("--reward_model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Path to reward model")
    parser.add_argument("--output_dir", type=str, default="/home/mila/j/jain.vineet/scratch/llada_checkpoints", help="Directory to save model")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU/CPU")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup steps ratio")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    
    # DeepSpeed arguments
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--ds_config", type=str, default="ds_config.json", help="DeepSpeed config path")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[2, 3], help="ZeRO optimization stage")
    
    # Logging and saving arguments
    parser.add_argument("--log_freq", type=int, default=1, help="Log metrics frequency")
    parser.add_argument("--save_steps", type=int, default=1000, help="Checkpoint frequency")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Max checkpoints to keep")
    
    # Dataset arguments
    parser.add_argument("--data_split", type=str, default="train", help="Training dataset split")
    parser.add_argument("--eval_split", type=str, default="test", help="Evaluation dataset split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # RWR specific arguments
    parser.add_argument("--replay_buffer_size", type=int, default=1024, help="Replay buffer size")
    parser.add_argument("--generation_batch_size", type=int, default=16, help="Generation batch size")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--generate_every", type=int, default=8, help="Sample generation frequency")
    parser.add_argument("--eval_freq", type=int, default=100, help="Evaluation frequency")
    
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

class ReplayBuffer:
    """FIFO replay buffer with uniform sampling for generated samples"""
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
    
    def add(self, sample):
        """Add a sample to the buffer in FIFO manner"""
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample)
        else:
            # Replace the oldest sample
            self.buffer[self.position] = sample
        
        # Update position for next insertion
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size):
        """Sample a batch of samples uniformly from the buffer"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Uniform sampling
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        return [self.buffer[idx] for idx in indices]
    
    def __len__(self):
        return len(self.buffer)

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

class RewardModel:
    def __init__(self, model_name_or_path):
        self.reward_model_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side="left")
        self.reward_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.device = torch.device(f"cuda:{dist.get_rank()}")
        self.reward_model.to(self.device)
        self.reward_model.eval()
        
    def compute_rewards(self, reference_questions: List[str], generated_responses: List[str], reference_answers: List[str]) -> List[float]:
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
        rewards = [1.0 if "YES" in response.strip().upper() else 0.0 for response in eval_responses]

        return rewards

class RWRTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{dist.get_rank()}")
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Load model and tokenizer
        logger.info(f"Loading model {args.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Load reward model
        logger.info(f"Loading reward model {args.reward_model_name_or_path}")
        self.reward_model = RewardModel(args.reward_model_name_or_path)
        
        # Load dataset
        logger.info("Loading GSM8K dataset")
        self.dataset = load_dataset('openai/gsm8k', 'main')[args.data_split]
        self.eval_dataset = load_dataset('openai/gsm8k', 'main')[args.eval_split]
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(max_size=args.replay_buffer_size)
        
        # Calculate total training steps
        num_update_steps_per_epoch = math.ceil(args.replay_buffer_size / (args.batch_size * args.grad_accum_steps))
        self.total_steps = args.num_train_epochs * num_update_steps_per_epoch
        
        # Update DeepSpeed config with computed values
        ds_config = self._prepare_deepspeed_config()
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            args=args,
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config
        )
        
        # Training tracking
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        
        logger.info(f"Rank {self.global_rank} initialized. World size: {self.world_size}")
        logger.info(f"Dataset size: {len(self.dataset)}, Steps per epoch: {num_update_steps_per_epoch}")
        logger.info(f"Effective batch size: {args.batch_size * self.world_size}, Total steps: {self.total_steps}")
    
    def _prepare_deepspeed_config(self):
        """Prepare DeepSpeed configuration."""
        if self.args.ds_config and os.path.exists(self.args.ds_config):
            with open(self.args.ds_config, "r") as f:
                ds_config = json.load(f)
            # Only override the ZeRO stage if specified
            if hasattr(self.args, 'zero_stage'):
                ds_config["zero_optimization"]["stage"] = self.args.zero_stage
            return ds_config
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
            if self.args.zero_stage == 3:
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
    
    def prepare_batch_for_gpu(self, batch, prepare_responses=False):
        """Prepare batch data for GPU processing."""
        questions = batch["question"]
        formatted_questions = []
        conversations = []
        responses = batch.get("response", []) if prepare_responses else []
        
        # Process each question separately
        for i, question in enumerate(questions):
            m = [{"role": "user", "content": question}]
            formatted = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            formatted_questions.append(formatted)
            
            if prepare_responses:
                conversations.append(formatted + responses[i] + '<|eot_id|>')
        
        prompt_encodings = self.tokenizer(formatted_questions)
        encodings = {"prompt_input_ids": self._pad_and_convert_to_tensor(prompt_encodings['input_ids'], MASK_ID)}
        # calculate prompt length
        
        
        if prepare_responses:
            conv_encodings = self.tokenizer(conversations)
            encodings["conv_input_ids"] = self._pad_and_convert_to_tensor(conv_encodings['input_ids'], self.tokenizer.eos_token_id)
            prompt_lengths = [len(item) for item in prompt_encodings['input_ids']]
            encodings["prompt_length"] = torch.tensor(prompt_lengths, dtype=torch.long, device=self.device)

        return encodings
    
    def _pad_and_convert_to_tensor(self, input_ids_list, pad_token_id):
        """Helper to pad a list of input IDs and convert to tensor on device."""
        max_length = max(len(item) for item in input_ids_list)
        
        for i in range(len(input_ids_list)):
            padding_length = max_length - len(input_ids_list[i])
            input_ids_list[i] = input_ids_list[i] + [pad_token_id] * padding_length
            
        return torch.tensor(input_ids_list).to(self.device)
    def generate_samples(self, dataset, num_samples=None):
        """Generate samples from the model for GSM8K problems"""
        self.model_engine.eval()
        
        if num_samples is None:
            num_samples = self.args.generation_batch_size * 32
        
        # Create a subset of indices to use
        dataset_size = len(dataset)
        indices = torch.randperm(dataset_size)[:num_samples].tolist()
        subset = torch.utils.data.Subset(dataset, indices)
        
        # Create a dataloader for more efficient batch processing
        sampler = DistributedSampler(
            subset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=True
        )

        dataloader = DataLoader(
            subset,
            batch_size=self.args.generation_batch_size,
            sampler=sampler,
            drop_last=True
        )
        
        samples = []
        for batch in tqdm(dataloader, desc="Generating samples", disable=self.global_rank != 0):
            # Prepare batch for GPU
            encodings = self.prepare_batch_for_gpu(batch)
            
            # Generate responses
            with torch.no_grad():
                outputs, _ = batch_generate(
                    self.model_engine, 
                    prompt=encodings["prompt_input_ids"], 
                    gen_length=256, 
                    block_length=32, 
                    steps=32, 
                    remasking='random', 
                    temperature=self.args.temperature
                )
            
            # Process generated responses
            batch_questions = []
            batch_responses = []
            batch_answers = []
            
            for idx, output in enumerate(outputs):
                # Determine the prompt length to extract only the generated content
                prompt_ids = encodings["prompt_input_ids"][idx]
                prompt_len = (prompt_ids != MASK_ID).sum().item()
                
                # Extract the generated response
                response_ids = output[prompt_len:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                
                batch_questions.append(batch["question"][idx])
                batch_responses.append(response)
                batch_answers.append(batch["answer"][idx])
            
            # Calculate rewards for the batch
            rewards = self.reward_model.compute_rewards(batch_questions, batch_responses, batch_answers)
            
            # Store samples with rewards
            for idx in range(len(batch_questions)):
                samples.append({
                    "question": batch_questions[idx],
                    "response": batch_responses[idx],
                    "reward": rewards[idx]
                })
                
                if len(samples) >= num_samples:  # Each GPU only needs its share
                    break
            
            if len(samples) >= num_samples:  # Each GPU only needs its share
                break
        
        self.model_engine.train()
        
        # Gather samples from all GPUs
        all_samples = [None for _ in range(self.world_size)]
        dist.all_gather_object(all_samples, samples)
        
        # Flatten the list of samples from all GPUs
        flat_samples = [sample for sublist in all_samples for sample in sublist if sample is not None]
        return flat_samples

    def train_step(self, batch):
        """Perform one training step."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model_engine.zero_grad()
        
        # Prepare batch for GPU
        encodings = self.prepare_batch_for_gpu(batch, prepare_responses=True)
        
        # Get rewards directly from the batch
        rewards = batch["reward"]
        
        # Forward pass and loss calculation
        loss = self.compute_loss(self.model_engine, encodings, rewards, self.device)
        
        # Backward pass
        self.model_engine.backward(loss)
        self.model_engine.step()

        # Ensure tensors are on the correct device for distributed operations
        loss = loss.clone().detach().to(self.device)
        rewards = torch.tensor(rewards, device=self.device)
        
        return loss, rewards.mean()

    def compute_loss(self, model, encodings, rewards, device, eps=1e-3):
        """
        Compute discrete diffusion loss for LLaDA model, weighted by rewards.
        """
        # Get batch size and sequence length
        input_ids = encodings["conv_input_ids"]
        prompt_lengths = encodings["prompt_length"]

        batch_size, seq_len = input_ids.shape
        
        # Generate random masking probability for each example in batch
        t = torch.rand(batch_size, device=device)
        # Scale probabilities between eps and 1-eps
        p_mask = (1 - eps) * t + eps
        # Expand to shape [batch_size, seq_len]
        p_mask = p_mask.unsqueeze(1).expand(-1, seq_len)
        
        # Create a mask for tokens that should be considered for masking
        # (only answer tokens should be masked, not prompt tokens)
        token_positions = torch.arange(seq_len, device=device).expand(batch_size, seq_len)
        prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
        
        # Generate random mask indices based on p_mask, but only for answer tokens
        rand_mask = torch.rand(batch_size, seq_len, device=device)
        # Ensure prompt tokens are never masked by setting their rand_mask values to > 1
        rand_mask.masked_fill_(prompt_mask, 2.0)
        masked_indices = rand_mask < p_mask
        
        # Create noisy batch by replacing masked tokens with mask_id
        noisy_batch = torch.where(masked_indices, MASK_ID, input_ids)
        
        # Double-check that prompt tokens are not masked
        noisy_batch = torch.where(prompt_mask, input_ids, noisy_batch)

        # Calculate the answer length for each example (including padded tokens)
        answer_lengths = seq_len - prompt_lengths
        
        # Forward pass through the model with noisy input
        model_outputs = model(noisy_batch)
        logits = model_outputs.logits
        
        # Only compute loss for the masked tokens
        # Extract logits and target tokens for masked positions
        masked_positions = (noisy_batch == MASK_ID)
        
        if masked_positions.sum() == 0:
            # If no tokens were masked, return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = input_ids.view(-1)
        
        # Extract values only for masked positions
        masked_logits = flat_logits[masked_positions.view(-1)]
        masked_targets = flat_targets[masked_positions.view(-1)]
        
        # Extract p_mask values for masked positions for loss weighting
        p_mask_flat = p_mask.reshape(-1)
        p_mask_masked = p_mask_flat[masked_positions.view(-1)]
        
        # Create a tensor of answer lengths that matches the shape of masked positions
        answer_lengths_expanded = answer_lengths.unsqueeze(1).expand(-1, seq_len)
        answer_lengths_flat = answer_lengths_expanded.reshape(-1)
        answer_lengths_masked = answer_lengths_flat[masked_positions.view(-1)]
        
        # Compute weighted cross-entropy loss for masked tokens
        token_loss = F.cross_entropy(masked_logits, masked_targets, reduction='none')
        
        # Weight loss by inverse mask probability and reward
        weighted_loss = token_loss / p_mask_masked
        normalized_loss = weighted_loss / answer_lengths_masked

        # Convert rewards to tensor and expand to match normalized_loss shape
        rewards_expanded = torch.tensor(rewards, device=device).unsqueeze(1).expand(-1, seq_len)
        rewards_flat = rewards_expanded.reshape(-1)
        rewards_masked = rewards_flat[masked_positions.view(-1)]
        
        # Compute final loss
        reward_weighted_loss = normalized_loss * rewards_masked
        
        # Average over batch
        loss = reward_weighted_loss.sum() / batch_size
        
        return loss

    def train(self):
        """Main training loop with DistributedSampler and periodic evaluation."""
        logger.info("Starting training...")
        if dist.get_rank() == 0:
            wandb.init(project="llada-rwr", config=vars(self.args))
        
        # Initial model samples
        logger.info("Generating initial samples for replay buffer...")
        initial_samples = self.generate_samples(self.dataset, self.replay_buffer_size)
        
        for sample in initial_samples:
            self.replay_buffer.add(sample)
        
        logger.info(f"Initial replay buffer size: {len(self.replay_buffer)}")
        
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            epoch_loss = 0
            steps_this_epoch = 0
            
            # Calculate steps per epoch based on current replay buffer size
            num_update_steps_per_epoch = math.ceil(len(self.replay_buffer) / (self.args.batch_size * self.args.grad_accum_steps))
            
            progress_bar = tqdm(range(num_update_steps_per_epoch), disable=self.global_rank != 0)
            
            for step in range(num_update_steps_per_epoch):
                # # Run evaluation at specified frequency
                # if self.global_step % self.args.eval_freq == 0:
                #     self.evaluate()
                
                # Sample from replay buffer
                batch_samples = self.replay_buffer.sample(self.args.batch_size * self.args.grad_accum_steps)
                
                # Skip if we don't have enough samples
                if len(batch_samples) < self.args.batch_size * self.args.grad_accum_steps:
                    logger.warning(f"Skipping step {step} due to insufficient samples in buffer (got {len(batch_samples)}, need {self.args.batch_size * self.args.grad_accum_steps})")
                    continue
                
                # Create batch dictionary
                batch = {
                    "question": [sample["question"] for sample in batch_samples],
                    "response": [sample["response"] for sample in batch_samples],
                    "reward": [sample["reward"] for sample in batch_samples]
                }
                
                # Accumulate gradients over multiple batches
                for i in range(self.args.grad_accum_steps):
                    start_idx = i * self.args.batch_size
                    end_idx = (i + 1) * self.args.batch_size
                    current_batch = {
                        "question": batch["question"][start_idx:end_idx],
                        "response": batch["response"][start_idx:end_idx],
                        "reward": batch["reward"][start_idx:end_idx]
                    }
                    
                    loss, rewards = self.train_step(current_batch)
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(rewards, op=dist.ReduceOp.AVG)
                    epoch_loss += loss
                    steps_this_epoch += 1
                    self.global_step += 1
                    
                    if self.global_rank == 0:
                        progress_bar.update(1)
                        progress_bar.set_description(f"Global Step {self.global_step}, Epoch {epoch}, Loss: {loss}")
                        
                        if self.global_step % self.args.log_freq == 0:
                            wandb.log({
                                "loss": loss.detach().cpu(),
                                "epoch": epoch,
                                "avg_reward": rewards.detach().cpu(),
                                "global_step": self.global_step,
                            })
                    
                    # Generate new samples periodically
                    if self.global_step % self.args.generate_every == 0:
                        logger.info("Generating new samples for replay buffer...")
                        # generate 32 batches of samples, this should probably be a separate hyperparameter
                        new_samples = self.generate_samples(self.dataset, self.args.generation_batch_size * 32)
                        
                        for sample in new_samples:
                            self.replay_buffer.add(sample)
                        
                        logger.info(f"Updated replay buffer size: {len(self.replay_buffer)}")
                    
                    if self.global_step % self.args.save_steps == 0:
                        # self.save_checkpoint()
                        pass
            
            avg_epoch_loss = epoch_loss / steps_this_epoch
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Run evaluation at the end of each epoch
            self.evaluate()

            # Generate new samples at end of epoch
            logger.info("Generating new samples for replay buffer...")
            # generate 32 batches of samples, this should probably be a separate hyperparameter
            new_samples = self.generate_samples(self.dataset, self.args.generation_batch_size * 32)
            
            for sample in new_samples:
                self.replay_buffer.add(sample)
            
            logger.info(f"Updated replay buffer size: {len(self.replay_buffer)}")
        
        logger.info("Training completed.")
        if dist.get_rank() == 0:
            wandb.finish()

    def evaluate(self):
        """Evaluate the model on the test set and report success rate (average reward)."""        
        logger.info(f"Running evaluation at step {self.global_step}...")
        
        # Switch to evaluation mode
        self.model_engine.eval()

        dataset_size = len(self.eval_dataset)
        indices = torch.randperm(dataset_size)[:256].tolist()
        subset = torch.utils.data.Subset(self.eval_dataset, indices)
        
        # Create a distributed sampler for the evaluation dataset
        eval_sampler = DistributedSampler(
            subset, 
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=True,
        )
        
        # Create a dataloader with the sampler
        eval_batch_size = 16  # Can use larger batch size for evaluation
        eval_dataloader = DataLoader(
            subset,
            batch_size=eval_batch_size,
            sampler=eval_sampler,
            drop_last=True,
        )
        
        # Track total and correct predictions
        total_rewards = []
        eval_steps = 0
        
        # Process batches
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=self.global_rank != 0):
            # Prepare batch for GPU
            encodings = self.prepare_batch_for_gpu(batch)
            
            # Generate responses
            with torch.no_grad():
                outputs, _ = batch_generate(
                    self.model_engine, 
                    prompt=encodings["prompt_input_ids"], 
                    gen_length=256, 
                    block_length=32, 
                    steps=32, 
                    remasking='random', 
                    temperature=0.0
                )
            
            # Process each sample in the batch
            responses = []
            for idx, output in enumerate(outputs):
                prompt_ids = encodings["prompt_input_ids"][idx]
                prompt_len = (prompt_ids != MASK_ID).sum().item()
                
                response_ids = output[prompt_len:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                responses.append(response)

            # Compute rewards
            rewards = self.reward_model.compute_rewards(
                batch["question"], 
                responses, 
                batch["answer"]
            )
            total_rewards.extend(rewards)
        
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
                "global_step": self.global_step
            })

            self.model_engine.train()
            return
        
        # Switch back to training mode
        self.model_engine.train()
        return

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
    trainer = RWRTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()
