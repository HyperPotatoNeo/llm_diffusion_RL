import os
import argparse
import logging
import math
import torch
import torch.nn.functional as F
import deepspeed
import numpy as np
import wandb
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    get_scheduler
)
from datasets import load_dataset
from dataloader import GSM8KDataset, collate_fn_factory, BlockGSM8KDataset, block_collate_fn_factory
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Fine-tuning of LLaDA-8B on GSM8K using DeepSpeed")
    
    # Model and dataset arguments
    parser.add_argument("--model_name_or_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--output_dir", type=str, default="/pscratch/sd/s/siddart2/llada_checkpoints",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--use_blocked_dataset", type=bool, choices=[True, False], default=False, help="Whether to use blocked dataset instead of full answer (for AR inference)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per GPU/CPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of update steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay to apply")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for gradient clipping")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="Learning rate scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of warmup steps to total training steps")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    
    # DeepSpeed arguments
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--deepspeed_config", type=str, default="gsm8k/ds_config.json",
                        help="Path to DeepSpeed configuration file")
    
    # Logging and saving arguments
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log training metrics every X steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="Save checkpoint every X steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")
    
    # Dataset arguments
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use for training")
    parser.add_argument("--eval_split", type=str, default="test",
                        help="Dataset split to use for evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    return args

def create_deepspeed_config(args):
    """Generate DeepSpeed config if not provided"""
    if os.path.exists(args.deepspeed_config):
        logger.info(f"Using DeepSpeed config from {args.deepspeed_config}")
        return
    
    logger.info("Creating default DeepSpeed configuration")
    config = {
        "train_batch_size": args.per_device_train_batch_size * torch.cuda.device_count() * args.gradient_accumulation_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": args.max_grad_norm,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "betas": [0.9, 0.999],
                "eps": 1e-8
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": 0,  # Will be set in main
                "total_num_steps": 0    # Will be set in main
            }
        },
        "bf16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }
    }
    
    with open(args.deepspeed_config, 'w') as f:
        import json
        json.dump(config, f, indent=4)
    
    logger.info(f"DeepSpeed configuration saved to {args.deepspeed_config}")

def compute_loss(model, batch, device, mask_id=126336, eps=1e-3):
    """
    Compute discrete diffusion loss for LLaDA model.
    This function masks random portions of the answer tokens (not prompt tokens)
    and calculates cross-entropy loss for those masked tokens.
    
    Args:
        model: The LLaDA model
        batch: A batch from the DataLoader containing conversation tokens and prompt lengths
        device: The device to run the computation on
        mask_id: Token ID for the mask token (default: 126336)
        eps: Minimum probability for masking (default: 1e-3)
    
    Returns:
        loss: The computed diffusion loss
    """
    # Get input_ids and attention_mask from batch
    input_ids = batch["conversation"]["input_ids"].to(device)
    attention_mask = batch["conversation"]["attention_mask"].to(device)
    prompt_lengths = batch["prompt_length"].to(device)
    
    # Get batch size and sequence length
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
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)
    
    # Double-check that prompt tokens are not masked
    noisy_batch = torch.where(prompt_mask, input_ids, noisy_batch)
    
    # Calculate the answer length for each example (including padded tokens)
    answer_lengths = seq_len - prompt_lengths
    
    # Forward pass through the model with noisy input
    outputs = model(noisy_batch)
    logits = outputs.logits
    
    # Only compute loss for the masked tokens
    # Extract logits and target tokens for masked positions
    masked_positions = (noisy_batch == mask_id)
    
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
    
    # Weight loss by inverse mask probability and normalize by answer length
    weighted_loss = token_loss / p_mask_masked
    normalized_loss = weighted_loss / answer_lengths_masked
    
    # Average over batch
    loss = normalized_loss.sum() / batch_size
    
    return loss

def main():
    args = parse_args()
    create_deepspeed_config(args)
    
    # Initialize distributed training
    deepspeed.init_distributed()
    
    # Initialize wandb only for the main process
    if args.local_rank <= 0:
        wandb.init(project="llada-sft", config=vars(args))
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model and tokenizer
    logger.info(f"Loading model {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Load dataset
    logger.info("Loading GSM8K dataset")
    gsm8k = load_dataset('openai/gsm8k', 'main')
    train_data = gsm8k[args.dataset_split]
    eval_data = gsm8k[args.eval_split]
    
    # Create datasets and dataloaders
    if args.use_blocked_dataset:
        train_dataset = BlockGSM8KDataset(train_data, tokenizer)
        eval_dataset = BlockGSM8KDataset(eval_data, tokenizer)
        collate_fn = block_collate_fn_factory(tokenizer)
    else:
        train_dataset = GSM8KDataset(train_data, tokenizer)
        eval_dataset = GSM8KDataset(eval_data, tokenizer)
        collate_fn = collate_fn_factory(tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    
    # Update DeepSpeed config with computed values
    if os.path.exists(args.deepspeed_config):
        with open(args.deepspeed_config, 'r') as f:
            import json
            ds_config = json.load(f)
            
        ds_config["scheduler"]["params"]["warmup_num_steps"] = int(max_train_steps * args.warmup_ratio)
        ds_config["scheduler"]["params"]["total_num_steps"] = max_train_steps
        
        with open(args.deepspeed_config, 'w') as f:
            json.dump(ds_config, f, indent=4)
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
        #config=args.deepspeed_config
    )
    
    # Training loop
    logger.info("***** Starting training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.epochs}")
    logger.info(f"  Per device batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    progress_bar = tqdm(range(max_train_steps), disable=not args.local_rank <= 0)
    completed_steps = 0
    best_eval_loss = float("inf")
    
    # Create output directory
    if args.local_rank <= 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model_engine.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            # Forward pass and loss calculation
            loss = compute_loss(model_engine, batch, model_engine.device)
            
            # Backward pass
            model_engine.backward(loss)
            
            # Update weights
            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                model_engine.step()
                completed_steps += 1
                progress_bar.update(1)
                
                # Log training progress
                total_loss += loss.item()
                if completed_steps % args.logging_steps == 0 and args.local_rank <= 0:
                    avg_loss = total_loss / args.logging_steps
                    logger.info(f"Epoch: {epoch}, Step: {completed_steps}, Loss: {avg_loss:.4f}")
                    wandb.log({"train_loss": avg_loss, "step": completed_steps, "epoch": epoch})
                    total_loss = 0
                
        # Evaluation at the end of each epoch
        model_engine.eval()
        eval_loss = 0
        eval_steps = 0
        
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not args.local_rank <= 0):
            with torch.no_grad():
                loss = compute_loss(model_engine, batch, model_engine.device)
                eval_loss += loss.item()
                eval_steps += 1
        
        eval_loss = eval_loss / eval_steps
        
        if args.local_rank <= 0:
            logger.info(f"Epoch {epoch} evaluation - Loss: {eval_loss:.4f}")
            wandb.log({"eval_loss": eval_loss, "epoch": epoch, "step": completed_steps})
            
            # Save if best model
            model_name = "llada_gsm8k_sft" if not args.use_blocked_dataset else "llada_blocked_gsm8k_sft"
            if eval_loss < best_eval_loss and args.local_rank <= 0:
                best_eval_loss = eval_loss
                output_dir = os.path.join(args.output_dir, model_name)
                logger.info("***** SAVING MODEL *****")
                os.makedirs(output_dir, exist_ok=True)
                
                # Save model with DeepSpeed
                model_engine.module.save_pretrained(output_dir)
                
                logger.info(f"New best model saved to {output_dir}")
    
    # Save final model
    if args.local_rank <= 0:
        logger.info("***** Training complete *****")
        wandb.finish()

if __name__ == "__main__":
    main()
