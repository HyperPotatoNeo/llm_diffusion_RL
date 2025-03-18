import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

@torch.no_grad()
def batch_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                   cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (batch_size, l) containing questions padded with mask_id to equal batch length.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Sampling temperature. If 0, uses argmax.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
    Returns:
        x: The final generated tensor.
        traj: A tensor containing, for each diffusion step, only the indices that were unmasked in that step,
              with shape (batch_size, num_diffusion_steps, final_sequence_length).
    '''
    batch_size, prompt_length = prompt.shape
    
    # Find the minimum question length (questions end at first mask token or are all the same length)
    mask_positions = (prompt == mask_id).nonzero(as_tuple=True)
    if len(mask_positions[0]) > 0:
        # Get the first mask position for each item in batch
        first_mask_indices = {}
        for batch_idx, seq_idx in zip(mask_positions[0].tolist(), mask_positions[1].tolist()):
            if batch_idx not in first_mask_indices or seq_idx < first_mask_indices[batch_idx]:
                first_mask_indices[batch_idx] = seq_idx
                
        # If some prompts don't have masks, use their full length
        min_question_length = prompt_length
        for batch_idx in range(batch_size):
            if batch_idx in first_mask_indices:
                min_question_length = min(min_question_length, first_mask_indices[batch_idx])
    else:
        # No masks in the prompts, they are all equal length
        min_question_length = prompt_length
    
    # Calculate the output length based on the maximum question length + gen_length
    output_length = prompt_length + gen_length

    # Create extended tensor with space for additional tokens
    x = torch.full((batch_size, output_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_length] = prompt.clone()
    
    # Create mask for the prompt (non-masked tokens)
    prompt_index = (x != mask_id)
    
    # Calculate the number of masks per sequence
    masks_per_sequence = torch.sum(x == mask_id, dim=1)
    
    # Calculate the maximum number of blocks needed across all sequences
    max_blocks_needed = (torch.max(masks_per_sequence).item() + block_length - 1) // block_length
    num_blocks = max_blocks_needed
    
    # Check if we have enough steps for all blocks
    steps_per_block = steps
    
    # Start generation from the minimum question length
    gen_start_idx = min_question_length
    
    # Initialize trajectory list. Each element will record only the tokens unmasked at that step.
    traj = []
    
    # Process each block sequentially
    for num_block in range(num_blocks):
        # Find positions of all mask tokens in each sequence, starting from gen_start_idx
        block_mask_indices = []
        for i in range(batch_size):
            # Get positions of mask tokens in the sequence from gen_start_idx onwards
            sequence_segment = x[i, gen_start_idx:].clone()
            mask_positions_seq = (sequence_segment == mask_id).nonzero().squeeze(-1)
            
            if len(mask_positions_seq) == 0:
                # No masks left to process in this sequence
                block_mask_index = torch.zeros_like(x[i], dtype=torch.bool)
                block_mask_indices.append(block_mask_index)
                continue
                
            # Calculate current block boundaries based on mask positions
            block_size = min(block_length, len(mask_positions_seq))
            end_pos_idx = min(block_size, len(mask_positions_seq))
            
            # Create mask tensor for this sequence's current block
            block_mask_index = torch.zeros_like(x[i], dtype=torch.bool)
            for pos_idx in range(end_pos_idx):
                abs_pos = gen_start_idx + mask_positions_seq[pos_idx].item()
                block_mask_index[abs_pos] = True
            block_mask_indices.append(block_mask_index)
        
        # Stack masks into batch
        block_mask_index = torch.stack(block_mask_indices)
        
        # Check if there are any masks left to process
        if not block_mask_index.any():
            break
        
        # Calculate the number of tokens to transfer each step
        masks_in_block = torch.sum(block_mask_index, dim=1)
        transfer_rate = masks_in_block / steps_per_block
        num_transfer_tokens = torch.ceil(transfer_rate).int()
        num_transfer_tokens = num_transfer_tokens.unsqueeze(1).expand(-1, steps_per_block)
        
        for i in range(steps_per_block):
            # Use the full sequence for model input
            mask_index = (x == mask_id)
            
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_cat = torch.cat([x, un_x], dim=0)
                logits = model(x_cat).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # Categorical sampling instead of gumbel softmax sampling
            if temperature == 0.:
                x0 = torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                # Reshape to sample one token per position for the whole batch
                x0 = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
                x0 = x0.view(batch_size, output_length)
            
            # Compute confidence for remasking
            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            # Only consider tokens within the current block for remasking
            confidence = torch.where(block_mask_index & mask_index, x0_p, -float('inf'))
            
            # For positions that are still masked, use the newly sampled x0
            x0 = torch.where(mask_index, x0, x)
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                if not confidence[j].gt(-float('inf')).any():
                    continue
                tokens_to_transfer = min(num_transfer_tokens[j, i].item(), (confidence[j] > -float('inf')).sum().item())
                if tokens_to_transfer > 0:
                    _, select_index = torch.topk(confidence[j], k=tokens_to_transfer)
                    transfer_index[j, select_index] = True
            
            # Update x with the newly unmasked tokens
            x[transfer_index] = x0[transfer_index]
            
            # Record only the tokens that were unmasked in this diffusion step.
            step_traj = torch.where(transfer_index, x, mask_id)
            traj.append(step_traj)
        
        # Update the starting point for the next block
        non_mask_count = torch.sum(block_mask_index, dim=1)
        if torch.all(non_mask_count == 0):
            break
        for batch_idx in range(batch_size):
            if non_mask_count[batch_idx] > 0:
                last_processed = torch.where(block_mask_index[batch_idx])[0][-1].item()
                gen_start_idx = max(gen_start_idx, last_processed + 1)
    
    # Convert the trajectory list to a tensor and rearrange dimensions:
    # from (num_steps, batch_size, final_sequence_length) to (batch_size, num_steps, final_sequence_length)
    traj_tensor = torch.stack(traj, dim=0).transpose(0, 1)
    
    return x, traj_tensor


@torch.no_grad()
def old_batch_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, eos_token_id=2):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (b, l) containing questions padded with mask_id to equal batch length.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        eos_token_id: The token id of EOS token to pad with (default: 2).
    '''
    batch_size, prompt_length = prompt.shape
    
    # Count existing mask tokens in each prompt
    mask_counts = torch.sum(prompt == mask_id, dim=1)  # shape: (batch_size,)
    
    # Calculate how many additional masks or EOS tokens to add for each sequence
    additional_masks = gen_length - mask_counts
    max_additional = torch.max(additional_masks).item()
    
    # Create extended tensor with space for additional tokens
    x = torch.full((batch_size, prompt_length + max_additional), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt_length] = prompt.clone()
    
    # For each sequence, add the correct number of mask tokens
    for i in range(batch_size):
        if additional_masks[i] > 0:
            # Add needed mask tokens
            x[i, prompt_length:prompt_length + additional_masks[i]] = mask_id
    
    # Create mask for the prompt (non-masked tokens)
    prompt_index = (x != mask_id)
    
    # Count total masks for each sequence (should be exactly gen_length for each)
    total_masks = torch.sum(x == mask_id, dim=1)
    assert torch.all(total_masks == gen_length), f"Not all sequences have {gen_length} masks: {total_masks}"

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        # Determine the starting index for the current block 
        # This is more complex now since mask tokens might be spread throughout the sequence
        block_start_indices = []
        block_end_indices = []
        
        for i in range(batch_size):
            # Find positions of all mask tokens in this sequence
            mask_positions = (x[i] == mask_id).nonzero().squeeze(-1)
            # Calculate block boundaries based on the mask positions
            block_size = mask_positions.size(0) // num_blocks
            start_idx = mask_positions[num_block * block_size].item()
            end_idx = mask_positions[min((num_block + 1) * block_size - 1, mask_positions.size(0) - 1)].item() + 1
            block_start_indices.append(start_idx)
            block_end_indices.append(end_idx)
        
        # Create block masks for each sequence
        block_mask_indices = []
        for i in range(batch_size):
            # Create a mask tensor for this sequence's current block
            sequence_mask = torch.zeros_like(x[i], dtype=torch.bool)
            sequence_mask[block_start_indices[i]:block_end_indices[i]] = (x[i, block_start_indices[i]:block_end_indices[i]] == mask_id)
            block_mask_indices.append(sequence_mask)
        
        # Stack masks into batch
        block_mask_index = torch.stack(block_mask_indices)
        
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)
            
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Prevent remasking tokens beyond the current block
            for j in range(batch_size):
                x0_p[j, :block_start_indices[j]] = -np.inf
                x0_p[j, block_end_indices[j]:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (b, l).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()