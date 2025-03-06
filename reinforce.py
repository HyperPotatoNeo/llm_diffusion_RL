import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel
from torch.distributions import Categorical
import torch.optim as optim
from generate import add_gumbel_noise, get_num_transfer_tokens
import random  # Added for random selection of grad steps
import wandb  # Added for wandb logging

class BatchREINFORCEDiffusion:
    def __init__(self, model_path='GSAI-ML/LLaDA-8B-Instruct', sentiment_model_path='cardiffnlp/twitter-roberta-base-sentiment-latest', 
                 device='cuda', learning_rate=1e-5):
        # Set device
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load discrete diffusion model
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.mask_id = 126336  # [MASK] token ID
        
        # Load sentiment reward model
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)
        self.sentiment_config = AutoConfig.from_pretrained(sentiment_model_path)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path).to(self.device)
        
        # Setup optimizer (only optimize the diffusion model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def preprocess_input(self, input_texts, batch_size=1):
        """Prepare the input for the model with batch support"""
        all_input_ids = []
        max_length = 0
        
        # Process each input text in the batch
        for input_text in input_texts:
            m = [{"role": "user", "content": input_text}]
            model_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            input_ids = self.tokenizer(model_input)['input_ids']
            all_input_ids.append(input_ids)
            max_length = max(max_length, len(input_ids))
        
        # Pad all sequences to the same length
        padded_input_ids = []
        for ids in all_input_ids:
            padded_ids = ids + [self.tokenizer.pad_token_id] * (max_length - len(ids))
            padded_input_ids.append(padded_ids)
        
        # Convert to tensor
        input_ids_tensor = torch.tensor(padded_input_ids).to(self.device)
        return input_ids_tensor
    
    def preprocess_sentiment_text(self, texts):
        """Preprocess texts for sentiment analysis (batch support)"""
        processed_texts = []
        for text in texts:
            new_text = []
            for t in text.split(" "):
                t = '@user' if t.startswith('@') and len(t) > 1 else t
                t = 'http' if t.startswith('http') else t
                new_text.append(t)
            processed_texts.append(" ".join(new_text))
        return processed_texts
    
    def evaluate_sentiment(self, texts):
        """Evaluate sentiment of generated texts, return negative sentiment scores"""
        processed_texts = self.preprocess_sentiment_text(texts)
        
        # Tokenize all texts in batch
        encoded_input = self.sentiment_tokenizer(processed_texts, return_tensors='pt', 
                                               padding=True, truncation=True).to(self.device)
        
        # Get sentiment scores
        with torch.no_grad():
            output = self.sentiment_model(**encoded_input)
        
        scores = output[0].detach().cpu().numpy()
        scores = softmax(scores, axis=1)
        
        # Return the negative sentiment scores (index 0 in the sentiment model)
        negative_scores = scores[:, 0]
        return negative_scores
    
    def reinforce_generation(self, prompts, batch_size, steps=128, gen_length=128, block_length=32, temperature=0.7, cfg_scale=0., grad_steps=128):
        """
        Generate text using REINFORCE learning with discrete diffusion (batch support).
        Only a random subset of grad_steps (out of total diffusion steps) will compute gradients.
        """
        input_ids = self.preprocess_input(prompts, batch_size)
        
        # Track log probs and selected tokens per batch item
        batch_log_probs = [[] for _ in range(batch_size)]
        
        # Start with masked sequences
        x = torch.full((batch_size, input_ids.shape[1] + gen_length), self.mask_id, dtype=torch.long).to(self.device)
        for b in range(batch_size):
            # Handle different prompt lengths
            prompt_length = (input_ids[b] != self.tokenizer.pad_token_id).sum().item()
            x[b, :prompt_length] = input_ids[b, :prompt_length].clone()
        
        # Create attention mask to identify actual prompt tokens (not padding)
        prompt_mask = (input_ids != self.tokenizer.pad_token_id)
        prompt_lengths = prompt_mask.sum(dim=1)
        
        # Create mask for prompt positions 
        prompt_index = torch.zeros_like(x, dtype=torch.bool)
        for b in range(batch_size):
            prompt_index[b, :prompt_lengths[b]] = True
        
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        # Compute total diffusion steps and select which steps will require grad
        total_steps = num_blocks * steps_per_block
        grad_steps_count = min(grad_steps, total_steps)
        grad_step_indices = sorted(random.sample(range(total_steps), grad_steps_count))
        global_step = 0
        
        self.model.train()  # Set model to training mode
        
        for num_block in range(num_blocks):
            # Define mask index for current block per batch item
            block_mask_indices = []
            for b in range(batch_size):
                prompt_len = prompt_lengths[b].item()
                block_start = prompt_len + num_block * block_length
                block_end = prompt_len + (num_block + 1) * block_length
                block_mask = (x[b, block_start:block_end] == self.mask_id)
                block_mask_indices.append(block_mask)
            
            # Convert to tensor
            block_mask_index = torch.stack(block_mask_indices)
            
            # Get number of tokens to transfer for each batch and step
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            
            for i in range(steps_per_block):
                mask_index = (x == self.mask_id)
                
                # Choose whether to track gradients this step or not
                if global_step in grad_step_indices:
                    # Forward pass with gradient tracking
                    if cfg_scale > 0.:
                        un_x = x.clone()
                        for b in range(batch_size):
                            un_x[b, prompt_index[b]] = self.mask_id
                        x_ = torch.cat([x, un_x], dim=0)
                        logits = self.model(x_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        # Debug print can be kept or removed as needed
                        print('HERE (grad mode): ', i)
                        logits = self.model(x).logits
                else:
                    # Forward pass without tracking gradients
                    with torch.no_grad():
                        if cfg_scale > 0.:
                            un_x = x.clone()
                            for b in range(batch_size):
                                un_x[b, prompt_index[b]] = self.mask_id
                            x_ = torch.cat([x, un_x], dim=0)
                            logits = self.model(x_).logits
                            logits, un_logits = torch.chunk(logits, 2, dim=0)
                            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                        else:
                            logits = self.model(x).logits
                
                # Apply temperature and create distribution
                logits_with_temp = logits / max(temperature, 1e-8)
                dist = Categorical(logits=logits_with_temp)
                x0 = dist.sample()  # Sample actions (tokens)
                
                # For remasking, compute confidence scores (using random remasking)
                x0_p = torch.rand(x0.shape, device=x0.device)
                
                # Prevent unmasking beyond the current block
                for b in range(batch_size):
                    prompt_len = prompt_lengths[b].item()
                    block_end = prompt_len + (num_block + 1) * block_length
                    x0_p[b, block_end:] = -np.inf
                
                # Update tokens based on sampling
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                
                # Identify which tokens will be transferred (unmasked) in this step
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for b in range(batch_size):
                    if num_transfer_tokens[b, i] > 0:
                        _, select_index = torch.topk(confidence[b], k=num_transfer_tokens[b, i])
                        transfer_index[b, select_index] = True
                
                # Calculate log probs only for tokens being unmasked in this step
                newly_unmasked = mask_index & transfer_index
                
                for b in range(batch_size):
                    newly_unmasked_positions = torch.nonzero(newly_unmasked[b]).squeeze()
                    if newly_unmasked_positions.numel() == 0:
                        continue
                    if newly_unmasked_positions.ndim == 0:
                        newly_unmasked_positions = newly_unmasked_positions.unsqueeze(0)
                    
                    newly_unmasked_log_probs = dist.log_prob(x0)[b, newly_unmasked_positions]
                    batch_log_probs[b].append(newly_unmasked_log_probs)
                
                # Update the tokens
                x = torch.where(transfer_index, x0, x)
                x = x.detach()
                global_step += 1
        
        # Return the generated sequences and log probs
        return x, batch_log_probs
    
    def compute_rewards(self, generated_sequences, prompt_lengths):
        """Compute the rewards based on sentiment score for each batch item"""
        generated_texts = []
        
        # Extract generated text for each batch item
        for b in range(len(generated_sequences)):
            prompt_len = prompt_lengths[b].item()
            generated_text = self.tokenizer.decode(
                generated_sequences[b, prompt_len:], 
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
        
        # Calculate sentiment scores - we want more negative sentiment
        negative_sentiment_scores = self.evaluate_sentiment(generated_texts)
        
        rewards = negative_sentiment_scores
        return rewards, generated_texts
    
    def update_policy(self, batch_log_probs, rewards):
        """Update the policy using REINFORCE algorithm with batch support"""
        policy_losses = []
        grad_batch_log_probs = []
        
        # Filter each batch item's log probs to only include those with gradients
        for lp in batch_log_probs:
            grad_tensors = [t for t in lp if t.requires_grad]
            if grad_tensors:  # Only if there is at least one tensor with grad
                grad_batch_log_probs.append(torch.cat(grad_tensors))
            else:
                grad_batch_log_probs.append(None)
        
        # Compute policy losses only for batches that have grad-tracked log probs
        for b, lp in enumerate(grad_batch_log_probs):
            if lp is not None:
                scaled_reward = rewards[b] - rewards.mean()  # Reward is a scalar (detached) constant
                policy_loss = -torch.mean(lp * scaled_reward)
                policy_losses.append(policy_loss)
        
        if policy_losses:
            total_loss = torch.mean(torch.stack(policy_losses))
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            return total_loss.item()
        else:
            return 0.0

    
    def train_step(self, prompts, batch_size=4, grad_steps=None):
        """
        Perform a single training step with batch support.
        Optionally, you can specify grad_steps. If not provided, all steps will compute gradients.
        """
        if len(prompts) < batch_size:
            prompts = prompts + [prompts[0]] * (batch_size - len(prompts))
        elif len(prompts) > batch_size:
            prompts = prompts[:batch_size]
        
        input_ids = self.preprocess_input(prompts, batch_size)
        prompt_mask = (input_ids != self.tokenizer.pad_token_id)
        prompt_lengths = prompt_mask.sum(dim=1)
        
        # Set grad_steps: if not provided, default to all steps (full grad)
        steps = 85
        if grad_steps is None:
            grad_steps = steps  # i.e. all steps require grad
        generated_sequences, batch_log_probs = self.reinforce_generation(
            prompts, 
            batch_size=batch_size,
            steps=85, 
            gen_length=85, 
            block_length=85, 
            temperature=0.5,
            grad_steps=grad_steps
        )
        
        rewards, generated_texts = self.compute_rewards(generated_sequences, prompt_lengths)
        loss = self.update_policy(batch_log_probs, rewards)
        
        return {
            'loss': loss,
            'rewards': rewards.tolist(),
            'generated_texts': generated_texts
        }
    
    def save_model(self, path):
        """Save the fine-tuned model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

def main():
    # Initialize wandb run with configuration settings
    wandb.init(project="llada-test", entity='swish', config={
        "num_episodes": 10000,
        "batch_size": 4,
        "grad_steps": 5,
        "learning_rate": 1e-5,
        "steps": 85,
        "gen_length": 85,
        "block_length": 85,
        "temperature": 0.5
    })

    # Initialize the Batch REINFORCE Diffusion agent
    agent = BatchREINFORCEDiffusion()
    
    # Training parameters
    num_episodes = wandb.config.num_episodes
    batch_size = wandb.config.batch_size
    # Set grad_steps to a fraction of total diffusion steps (e.g., if steps=64 and you want 25% grad steps, use 16)
    grad_steps = wandb.config.grad_steps
    
    prompts = [
        "Write a small story about a man walking in the rain"
    ]
    
    print(f"Starting Batch REINFORCE training for negative review generation (batch size: {batch_size})...")
    
    for episode in range(num_episodes):
        results = agent.train_step(prompts, batch_size=batch_size, grad_steps=grad_steps)
        
        # Log metrics to wandb
        wandb.log({
            "episode": episode + 1,
            "loss": results['loss'],
            "mean_reward": np.mean(results['rewards']),
            "min_reward": np.min(results['rewards']),
            "max_reward": np.max(results['rewards']),
            #"sample_generated_text": results['generated_texts'][0] if results['generated_texts'] else ""
        })
        if episode == 0:
            table = wandb.Table(columns=["episode", "sample_generated_text"])
        table.add_data(episode + 1, results['generated_texts'][0] if results['generated_texts'] else "")
        wandb.log({"samples_table": table})
        
        print(f"Episode {episode+1}/{num_episodes}")
        print(f"Loss: {results['loss']:.4f}")
        for i, (reward, text) in enumerate(zip(results['rewards'], results['generated_texts'])):
            print(f"  Batch {i+1}: Reward: {reward:.4f}")
            print(f"  Text: {text[:100]}...")
        print("-" * 50)
        
        #if (episode + 1) % 5 == 0:
        #    agent.save_model(f"llada-negative-reviews-batch-episode-{episode+1}")
    
    agent.save_model("llada-negative-reviews-batch-final")
    print("Training completed!")
    
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
