import json
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random

class BlockGSM8KDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, block_size=32, max_answer_length=256):
        """
        Args:
            data: Train or test dataset.
            tokenizer: Tokenizer instance that implements apply_chat_template.
            max_length (int): Maximum token length for truncation of the prompt.
            block_size (int): The size of an autoregressive block (default: 32 tokens).
            max_answer_length (int): Maximum answer tokens (folded + target) retained (default: 256 tokens).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.block_size = block_size
        self.max_answer_length = max_answer_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample["question"].strip()
        answer = sample["answer"].strip()
        
        # Build the initial prompt portion (from the question).
        orig_prompt = self.tokenizer.apply_chat_template([
            {"role": "user", "content": question},
        ], tokenize=False, add_generation_prompt=True)
        
        # Tokenize the answer (without adding special tokens) and truncate it to max_answer_length.
        answer_tokens = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        answer_tokens = answer_tokens[:self.max_answer_length]
        L = len(answer_tokens)
        
        # Determine how many full blocks to fold into the prompt.
        # We want to reserve one block (self.block_size) as the target.
        if L >= self.block_size:
            max_foldable = L - self.block_size  # tokens that can be folded.
            # Choose a number of complete blocks (n) from 0 up to the max possible.
            possible_fold_chunks = max_foldable // self.block_size
            folded_chunks = random.randint(0, possible_fold_chunks) if possible_fold_chunks > 0 else 0
        else:
            folded_chunks = 0
        folded_length = folded_chunks * self.block_size
        
        # Get the target block: the next block_size tokens after the folded portion.
        target_tokens = answer_tokens[folded_length: folded_length + self.block_size]
        # Pad target_tokens with EOS if not enough tokens.
        if len(target_tokens) < self.block_size:
            pad_length = self.block_size - len(target_tokens)
            target_tokens = target_tokens + [self.tokenizer.eos_token_id] * pad_length

        # Fold the chosen portion of the answer into the prompt.
        folded_tokens = answer_tokens[:folded_length]
        # Decode the folded tokens back to string (preserving special tokens as needed).
        folded_text = self.tokenizer.decode(folded_tokens, skip_special_tokens=False)
        new_prompt = orig_prompt + folded_text
        
        # Tokenize the new prompt.
        prompt_tokens = self.tokenizer(
            new_prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        prompt_length = len(prompt_tokens["input_ids"])
        prompt_tokens["input_ids"] = prompt_tokens["input_ids"] + target_tokens
        
        return {
            "input_ids": prompt_tokens["input_ids"],        # New prompt: question + folded answer tokens.
            "attention_mask": prompt_tokens["attention_mask"],
            "prompt_length": prompt_length,                   # Length of the new prompt.
        }

def block_collate_fn_factory(tokenizer):
    """
    Returns a collate function that pads the prompt tokens to the same length across the batch.
    Also, it returns the target block tokens.
    """
    eos_token_id = tokenizer.eos_token_id

    def collate_fn(batch):
        # Get tokenized prompt sequences, their attention masks, and target tokens.
        input_ids_list = [item["input_ids"] for item in batch]
        attn_mask_list = [item["attention_mask"] for item in batch]
        prompt_lengths = [item["prompt_length"] for item in batch]
        
        # Determine the maximum length for padding the prompt.
        max_len = max(len(ids) for ids in input_ids_list)
        
        # Pad each prompt with EOS tokens and attention masks with 0.
        padded_input_ids = [
            ids + [eos_token_id] * (max_len - len(ids))
            for ids in input_ids_list
        ]
        padded_attn_masks = [
            mask + [0] * (max_len - len(mask))
            for mask in attn_mask_list
        ]
        
        conversation = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn_masks, dtype=torch.long)
        }
        prompt_lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.long)
        
        return {
            "conversation": conversation,
            "prompt_length": prompt_lengths_tensor
        }
    
    return collate_fn

class GSM8KDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        """
        Args:
            data (str): Train or test dataset
            tokenizer: Tokenizer instance that implements apply_chat_template.
            max_length (int): Maximum token length for truncation.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        question = sample["question"].strip()
        answer = sample["answer"].strip()
        
        # Build the prompt portion only (includes "Assistant:" but no answer).
        prompt = self.tokenizer.apply_chat_template([
            {"role": "user", "content": question},
        ], tokenize=False, add_generation_prompt=True)
        # The tokenizer chat template with both user and assistant seems to have some bug, and adds another assistant header at end
        # Explicitly adding eot token at end of answer to finish the turn
        conversation = prompt + answer + '<|eot_id|>'

        # Tokenize the full conversation.
        conv_tokens = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        
        # Tokenize the prompt to compute its token length.
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )
        prompt_length = len(prompt_tokens["input_ids"])
        
        return {
            "input_ids": conv_tokens["input_ids"],        # Full conversation tokens.
            "attention_mask": conv_tokens["attention_mask"],
            "prompt_length": prompt_length                  # Token count for prompt only.
        }

def collate_fn_factory(tokenizer):
    """
    Returns a collate function that pads the concatenated conversation tokens
    with EOS tokens so that every sample in the batch has the same length.
    Also, it returns the prompt lengths.
    """
    eos_token_id = tokenizer.eos_token_id

    def collate_fn(batch):
        # Get tokenized conversation sequences and prompt lengths.
        input_ids_list = [item["input_ids"] for item in batch]
        attn_mask_list = [item["attention_mask"] for item in batch]
        prompt_lengths = [item["prompt_length"] for item in batch]
        
        # Determine the maximum length for padding.
        max_len = max(len(ids) for ids in input_ids_list)
        
        # Pad each sequence with EOS tokens and pad attention masks with 0.
        padded_input_ids = [
            ids + [eos_token_id] * (max_len - len(ids))
            for ids in input_ids_list
        ]
        padded_attn_masks = [
            mask + [0] * (max_len - len(mask))
            for mask in attn_mask_list
        ]
        
        # Convert lists to tensors.
        conversation = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn_masks, dtype=torch.long)
        }
        #print(tokenizer.batch_decode(conversation['input_ids'])[0])
        prompt_lengths_tensor = torch.tensor(prompt_lengths, dtype=torch.long)
        
        return {
            "conversation": conversation,
            "prompt_length": prompt_lengths_tensor
        }
    
    return collate_fn

# Example usage:
if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', 
                                            trust_remote_code=True)
    
    print("Loading GSM8K test dataset from Hugging Face...")
    gsm8k = load_dataset('openai/gsm8k', 'main')
    train_data = gsm8k["train"]
    test_data = gsm8k["test"]
    
    dataset = GSM8KDataset(test_data, tokenizer, max_length=512)
    collate_fn = collate_fn_factory(tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Iterate through one batch to inspect its shapes.
    for batch in dataloader:
        print(tokenizer.batch_decode(batch["conversation"]["input_ids"]))
        print("Conversation input_ids shape:", batch["conversation"]["input_ids"].shape)
        print("Conversation attention_mask shape:", batch["conversation"]["attention_mask"].shape)
        print("Prompt lengths:", batch["prompt_length"])
        break