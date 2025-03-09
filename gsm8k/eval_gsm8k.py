import torch
import json
import re
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import deepspeed
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# Import your generate function
from generate import batch_generate

class GSM8KDataset(Dataset):
    def __init__(self, questions, tokenizer, mask_id=126336):
        self.questions = questions
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        m = [{"role": "user", "content": question}]
        model_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        tokens = self.tokenizer(model_input)['input_ids']
        return {"input_ids": tokens, "idx": idx}

def collate_fn(batch, mask_id=126336):
    # Find the maximum input length in this batch
    max_length = max(len(item["input_ids"]) for item in batch)
    
    # Right-pad all inputs to the same length
    padded_inputs = []
    indices = []
    
    for item in batch:
        tokens = item["input_ids"]
        idx = item["idx"]
        padding_length = max_length - len(tokens)
        # Right pad with mask_id
        padded_tokens = tokens + [mask_id] * padding_length
        padded_inputs.append(padded_tokens)
        indices.append(idx)
    
    # Convert to tensor
    batch_input_ids = torch.tensor(padded_inputs)
    indices = torch.tensor(indices)
    
    return {"input_ids": batch_input_ids, "indices": indices}

def extract_answer(text):
    """
    Extract the final answer from generated text.
    The model should output the final answer after ####.
    """
    # Find the last occurrence of ####
    match = re.search(r'####\s*([\d\.\-\+]+)', text)
    if match:
        try:
            # Extract the numeric answer
            return float(match.group(1).strip())
        except ValueError:
            return None
    return None

def batch_generate_deepspeed(model, dataloader, tokenizer, gen_length=256, steps=256, block_length=256, temperature=0.7):
    """
    Generate answers using DeepSpeed for multi-GPU processing.
    Instead of preallocating a list (which assumed global indexing), we store
    the generated text in a dictionary mapping the global sample index to the output.
    """
    local_generated = {}
    
    for batch in tqdm(dataloader, desc="Processing Batches"):
        batch_input_ids = batch["input_ids"].to(model.device)
        indices = batch["indices"]
        
        # Generate text in batch
        out = batch_generate(
            model, 
            batch_input_ids, 
            steps=steps, 
            gen_length=gen_length, 
            block_length=block_length, 
            temperature=temperature, 
            cfg_scale=0., 
            remasking='low_confidence',
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Process generated text and store results using the global index
        for i, idx in enumerate(indices.tolist()):
            generated_text = tokenizer.decode(out[i, :], skip_special_tokens=True)
            local_generated[idx] = generated_text
    return local_generated

def evaluate_gsm8k_deepspeed(model_engine, tokenizer, config_path, batch_size=4):
    # Mask token ID
    mask_id = 126336
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    if rank == 0:
        print("Starting evaluation on GSM8K dataset...")
    
    # Generation Parameters
    gen_length = 256  # Longer generation for math problems with reasoning
    steps = 256
    block_length = 256
    temperature = 0.0
    
    # Load GSM8K test dataset (each process loads the full dataset)
    if rank == 0:
        print("Loading GSM8K test dataset from Hugging Face...")
    gsm8k = load_dataset('openai/gsm8k', 'main')
    test_data = list(gsm8k["test"])
    
    # Prepare lists to store all questions and reference answers (global lists)
    questions = []
    reference_answers = []
    ground_truths = []
    
    if rank == 0:
        print("Processing dataset...")
    for item in test_data:
        question = item["question"]
        reference_answer = item["answer"]
        
        # Extract the ground truth numeric answer
        gt_match = re.search(r'####\s*([\d\.\-\+]+)', reference_answer)
        if not gt_match:
            continue
        
        ground_truth = float(gt_match.group(1).strip())
        
        questions.append(question)
        reference_answers.append(reference_answer)
        ground_truths.append(ground_truth)
    
    # Create dataset and use a DistributedSampler so that each process gets a unique subset.
    dataset = GSM8KDataset(questions, tokenizer, mask_id)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=lambda batch: collate_fn(batch, mask_id)
    )
    
    if rank == 0:
        print(f"Generating answers with batch size {batch_size} on {world_size} device(s)...")
    
    # Generate answers on each process; local_generated maps global indices to outputs.
    local_generated = batch_generate_deepspeed(
        model_engine,
        dataloader,
        tokenizer,
        gen_length=gen_length, 
        steps=steps, 
        block_length=block_length, 
        temperature=temperature
    )
    
    # Gather dictionaries from all processes so that rank 0 can merge and evaluate.
    gathered_generated = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_generated, local_generated)
    
    # Only the main process aggregates and evaluates the full results.
    if rank == 0:
        # Merge the dictionaries
        global_generated = {}
        for d in gathered_generated:
            global_generated.update(d)
        
        print("Evaluating answers...")
        total = 0
        correct = 0
        results = []
        
        # Iterate over the sorted global indices to maintain original order
        for idx in sorted(global_generated.keys()):
            generated_text = global_generated[idx]
            question = questions[idx]
            ground_truth = ground_truths[idx]
            
            # Extract the predicted answer
            predicted_answer = extract_answer(generated_text)
            
            # Check if answer is correct (using a tolerance for floating point comparisons)
            is_correct = False
            if predicted_answer is not None:
                is_correct = abs(predicted_answer - ground_truth) < 1e-6
                
            total += 1
            if is_correct:
                correct += 1
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "model_output": generated_text,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct
            })
            
            # Optionally, save intermediate results every 10 examples
            if (total) % 10 == 0:
                with open("gsm8k_deepspeed_results.json", "w") as f:
                    json.dump({
                        "accuracy": correct / total if total > 0 else 0,
                        "correct": correct,
                        "total": total,
                        "results": results
                    }, f, indent=2)
        
        # Calculate final accuracy
        accuracy = correct / total if total > 0 else 0
        
        # Save final results
        with open("gsm8k_deepspeed_results.json", "w") as f:
            json.dump({
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results
            }, f, indent=2)
        
        print(f"\nEvaluation complete!")
        print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return accuracy, results
    
    return None, None

if __name__ == "__main__":
    # Create a single argument parser
    parser = argparse.ArgumentParser(description="Evaluate LLaDA model on GSM8K dataset using DeepSpeed")
    
    # Add DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)
    
    # Add our custom arguments
    parser.add_argument("--config", type=str, default="gsm8k/ds_config.json", 
                        help="DeepSpeed configuration file")
    parser.add_argument("--batch_size", type=int, default=5, 
                        help="Batch size per GPU")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    args = parser.parse_args()
    
    # Initialize distributed environment
    deepspeed.init_distributed()
    
    # Load model
    if torch.distributed.get_rank() == 0:
        print("Loading LLaDA-8B-Instruct model...")
    
    model = AutoModel.from_pretrained('/pscratch/sd/s/siddart2/llada_checkpoints/llada_gsm8k_sft', 
                                      trust_remote_code=True, 
                                      torch_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', 
                                             trust_remote_code=True)
    
    # Initialize DeepSpeed
    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=args.config
    )
    model_engine.eval()
    
    # Run evaluation
    evaluate_gsm8k_deepspeed(model_engine, tokenizer, args.config, batch_size=args.batch_size)
