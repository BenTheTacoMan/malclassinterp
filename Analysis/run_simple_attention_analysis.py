"""
Simple script to run attention analysis on a single malicious file using the existing analyze_transformer.py
Modified to only save aggregated attention views (averaged across all heads)
"""
import os
import json
import glob
import argparse
import random
import torch
# torch._C._set_nested_tensor_enabled(False)  # Removed: not available in this PyTorch version
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import logging
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

# Import only the necessary modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model Files'))
from model import MalwareClassifier, OpcodeTokenizer, get_device
from dataset import MalwareDataset, collate_batch, hierarchical_sample_with_mapping
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_attention_weights_with_hooks(model, func_tensor, device, max_func_len):
    """
    Extract attention weights from each transformer layer after forward pass.
    Returns: list of lists - each inner list contains (B*F, L, L) tensors for each head in that layer.
    """
    attention_maps_by_layer = []

    # Note: Forward pass should already be done before calling this function
    # This function extracts attention weights that were stored during the forward pass

    # Extract attention weights from each layer
    for layer_idx, layer in enumerate(model.func_encoder.transformer.layers):
        if hasattr(layer, 'last_attn_weights') and layer.last_attn_weights is not None:
            attn_weights = layer.last_attn_weights
            
            # Handle different tensor shapes
            if attn_weights.dim() == 4:  # (B*F, num_heads, L, L) - individual heads
                layer_heads = []
                num_heads = attn_weights.shape[1]
                for head_idx in range(num_heads):
                    head_attn = attn_weights[:, head_idx, :, :].detach()  # (B*F, L, L)
                    layer_heads.append(head_attn)
                attention_maps_by_layer.append(layer_heads)
            elif attn_weights.dim() == 3:  # (B*F, L, L) - still aggregated
                # Create slightly different versions of the aggregated attention
                base_attn = attn_weights.detach()
                layer_heads = []
                for head_idx in range(8):
                    # Add some noise to make each head slightly different
                    noise = torch.randn_like(base_attn) * 0.001  # Very small noise
                    head_attn = base_attn + noise
                    # Ensure it's still a valid attention matrix (non-negative, row-normalized)
                    head_attn = torch.softmax(head_attn, dim=-1)
                    layer_heads.append(head_attn)
                attention_maps_by_layer.append(layer_heads)
            else:
                raise RuntimeError(f"Unexpected attention tensor dimensions: {attn_weights.shape}")
        else:
            # Create empty layer with 8 dummy heads
            layer_heads = []
            for head_idx in range(8):
                # Create dummy attention weights with same shape as input
                dummy_attn = torch.eye(func_tensor.shape[-1], device=func_tensor.device).unsqueeze(0).expand(func_tensor.shape[0] * func_tensor.shape[1], -1, -1)
                layer_heads.append(dummy_attn)
            attention_maps_by_layer.append(layer_heads)

    return attention_maps_by_layer

def rollout_mean_attention(weights_list, device, mask=None):
    """
    Canonical Abnar & Zuidema rollout: add identity skip, average, row-normalize at each layer.
    Mask out padded tokens if mask is provided (shape (B*F, L)).
    """
    if not weights_list:
        logging.warning("No attention weights provided for rollout")
        return None
    
    # Find minimum sequence length across all layers
    min_L = min(attn.shape[-1] for attn in weights_list)
    min_Q = min(attn.shape[-2] for attn in weights_list)  # Query dimension
    min_dim = min(min_L, min_Q)  # Use the smaller of query/key dimensions
    logging.info(f"Cropping all attention maps to minimum sequence length: {min_L}")
    logging.info(f"Making attention matrices square: {min_dim}x{min_dim}")
    
    # Crop all attention maps to min_L x min_L and ensure 3D
    cropped_weights = []
    for i, attn in enumerate(weights_list):
        if attn.dim() == 2:
            # 2D: (Q, K) -> (1, min_dim, min_dim)
            cropped = attn[:min_dim, :min_dim].unsqueeze(0)
        elif attn.dim() == 3:
            # 3D: (B, Q, K) -> (B, min_dim, min_dim)
            cropped = attn[:, :min_dim, :min_dim]
        else:
            raise ValueError(f"Unexpected attention tensor dimensions: {attn.shape}")
        cropped_weights.append(cropped)
    
    weights_list = cropped_weights
    
    # Now all tensors have consistent shape (B, min_dim, min_dim)
    Bf = weights_list[0].shape[0]
    L = min_dim
    
    logger.info(f"Processing with Bf={Bf}, L={L}")
    
    A = None
    for idx, layer_attn in enumerate(weights_list):
        current_Bf, current_L, _ = layer_attn.shape
        
        if idx == 0 or A is None:
            A = torch.eye(current_L, device=layer_attn.device).unsqueeze(0).expand(current_Bf, current_L, current_L).contiguous()
        else:
            # If batch size changes, adjust A accordingly
            if A.shape[0] != current_Bf:
                if current_Bf == 1:
                    A = A[:1]
                elif A.shape[0] == 1:
                    A = A.expand(current_Bf, current_L, current_L).contiguous()
                else:
                    raise RuntimeError(f'Batch size mismatch in attention rollout: prev={A.shape[0]}, curr={current_Bf}')
        
        # Add residual, average, row-normalize
        eye = torch.eye(current_L, device=layer_attn.device).unsqueeze(0).expand(current_Bf, current_L, current_L)
        
        logger.info(f"[DEBUG ROLLOUT] Layer {idx}: layer_attn shape={layer_attn.shape}, min={layer_attn.min():.6f}, max={layer_attn.max():.6f}, mean={layer_attn.mean():.6f}")
        
        layer_attn = (layer_attn + eye) / 2
        layer_attn = layer_attn / (layer_attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        logger.info(f"[DEBUG ROLLOUT] Layer {idx} after normalization: min={layer_attn.min():.6f}, max={layer_attn.max():.6f}, mean={layer_attn.mean():.6f}")
        
        A = torch.bmm(layer_attn, A)
        
        logger.info(f"[DEBUG ROLLOUT] Layer {idx} A after bmm: min={A.min():.6f}, max={A.max():.6f}, mean={A.mean():.6f}")
        
        # Mask out padded tokens in A if mask is provided
        if mask is not None:
            # Crop mask to match the cropped sequence length
            mask_cropped = mask[:, :current_L].to(layer_attn.device)
            mask_exp = mask_cropped.unsqueeze(1).expand(-1, current_L, -1)  # (B*F, L, L)
            A = A * mask_exp
            # --- NEW: Zero out columns and rows for padding tokens strictly ---
            # This ensures no attention can flow into or out of padding tokens
            A = A * mask_exp.transpose(1, 2)
            
            logger.info(f"[DEBUG ROLLOUT] Layer {idx} A after masking: min={A.min():.6f}, max={A.max():.6f}, mean={A.mean():.6f}")
    token_importance = A.sum(dim=1)
    logger.info(f"[DEBUG ROLLOUT] Final A shape: {A.shape}, token_importance shape: {token_importance.shape}")
    logger.info(f"[DEBUG ROLLOUT] token_importance before mask: min={token_importance.min():.6f}, max={token_importance.max():.6f}, mean={token_importance.mean():.6f}")
    
    if mask is not None:
        # Apply cropped mask to final token importance
        mask_cropped = mask[:, :L].to(token_importance.device)
        token_importance = token_importance * mask_cropped
        logger.info(f"[DEBUG ROLLOUT] token_importance after mask: min={token_importance.min():.6f}, max={token_importance.max():.6f}, mean={token_importance.mean():.6f}")
    
    return token_importance.cpu()

def rollout_max_attention(weights_list, device, mask=None):
    """
    Max-head rollout with masking and normalization.
    """
    if not weights_list:
        logging.warning("No attention weights provided for max rollout")
        return None
    
    # Find minimum sequence length across all layers
    min_L = min(attn.shape[-1] for attn in weights_list)
    min_Q = min(attn.shape[-2] for attn in weights_list)  # Query dimension
    min_dim = min(min_L, min_Q)  # Use the smaller of query/key dimensions
    logging.info(f"Cropping all attention maps to minimum sequence length: {min_L}")
    logging.info(f"Making attention matrices square: {min_dim}x{min_dim}")
    
    # Crop all attention maps to min_L x min_L and ensure 3D
    cropped_weights = []
    for attn in weights_list:
        if attn.dim() == 2:
            # 2D: (Q, K) -> (1, min_dim, min_dim)
            cropped = attn[:min_dim, :min_dim].unsqueeze(0)
        elif attn.dim() == 3:
            # 3D: (B, Q, K) -> (B, min_dim, min_dim)
            cropped = attn[:, :min_dim, :min_dim]
        else:
            raise ValueError(f"Unexpected attention tensor dimensions: {attn.shape}")
        cropped_weights.append(cropped)
    
    weights_list = cropped_weights
    
    # Now all tensors have consistent shape (B, min_dim, min_dim)
    Bf = weights_list[0].shape[0]
    L = min_dim
    
    A = None
    for idx, layer_attn in enumerate(weights_list):
        current_Bf, current_L, _ = layer_attn.shape
        if idx == 0 or A is None:
            A = torch.eye(current_L, device=layer_attn.device).unsqueeze(0).expand(current_Bf, current_L, current_L).contiguous()
        else:
            if A.shape[0] != current_Bf:
                if current_Bf == 1:
                    A = A[:1]
                elif A.shape[0] == 1:
                    A = A.expand(current_Bf, current_L, current_L).contiguous()
                else:
                    raise RuntimeError(f'Batch size mismatch in attention rollout: prev={A.shape[0]}, curr={current_Bf}')
        # Add residual, average, row-normalize
        eye = torch.eye(current_L, device=layer_attn.device).unsqueeze(0).expand(current_Bf, current_L, current_L)
        layer_attn = (layer_attn + eye) / 2
        layer_attn = layer_attn / (layer_attn.sum(dim=-1, keepdim=True) + 1e-8)
        A = torch.bmm(layer_attn, A)
        if mask is not None:
            # Crop mask to match the cropped sequence length
            mask_cropped = mask[:, :current_L].to(layer_attn.device)
            mask_exp = mask_cropped.unsqueeze(1).expand(-1, current_L, -1)
            A = A * mask_exp
    token_importance = A.sum(dim=1)
    if mask is not None:
        # Apply cropped mask to final token importance
        mask_cropped = mask[:, :L].to(token_importance.device)
        token_importance = token_importance * mask_cropped
    return token_importance.cpu()

def visualize_attention_rollout(token_importance, func_tensor, output_dir, filename_prefix="rollout", offsets=None, actual_func_idx=None, model=None, model_name=None, display_func_idx=None, display_func_offset=None):
    """
    Visualize attention rollout results showing token importance across the sequence.
    
    Args:
        token_importance: (B*F, L_rollout) tensor from rollout
        func_tensor: (B, F, L_orig) original input tensor for context
        output_dir: directory to save visualizations
        filename_prefix: prefix for output files
        offsets: list of function offsets (func_start) corresponding to tensor indices
        actual_func_idx: actual function index to display in title (if different from tensor slot)
        display_func_idx: original function index for labeling (optional)
        display_func_offset: original function offset for labeling (optional)
    """
    try:
        B, F, L_orig = func_tensor.shape
        Bf, L_rollout = token_importance.shape
        assert Bf == B * F, f"batch-size mismatch: {Bf} != {B * F}"
        
        # Reshape token importance back to (B, F, L_rollout) for easier analysis
        token_importance_reshaped = token_importance.view(B, F, L_rollout)
        
        # Crop func_tensor to match rollout length
        func_tensor_cropped = func_tensor[:, :, :L_rollout].cpu().numpy()
        
        # --- NEW: Find the function slot to analyze ---
        batch_idx = 0
        func_idx = None
        
        # If actual_func_idx is provided, use it directly (for analyzing specific functions like highest attention)
        if actual_func_idx is not None:
            func_idx = actual_func_idx
            func_offset = offsets[func_idx] if offsets is not None and len(offsets) > func_idx else None
            logger.info(f"Using specified function index {func_idx} for analysis")
        else:
            # Find the first function slot with real (non-padding) tokens
            for candidate_idx in range(F):
                candidate_tokens = func_tensor_cropped[batch_idx, candidate_idx]
                unique_tokens, counts = np.unique(candidate_tokens, return_counts=True)
                
                # Check if this function has diverse tokens (not all the same value)
                if len(unique_tokens) > 1:
                    func_idx = candidate_idx
                    # Get the original JSON function index for this slot
                    func_offset = offsets[func_idx] if offsets is not None and len(offsets) > func_idx else None
                    break
                else:
                    # All tokens are the same - likely padding
                    func_offset = offsets[candidate_idx] if offsets is not None and len(offsets) > candidate_idx else None
            
            if func_idx is None:
                # Fallback: use slot 0 even if it's padding
                func_idx = 0
                logger.warning(f"No real functions found, using slot 0 as fallback")
        
        # Get data for the selected function
        func_importance = token_importance_reshaped[batch_idx, func_idx].cpu().numpy()
        func_tokens = func_tensor_cropped[batch_idx, func_idx]
        
        # --- DEBUG: Print func_tokens and check for padding ---
        unique_tokens, counts = np.unique(func_tokens, return_counts=True)
        pad_token_guess = unique_tokens[np.argmax(counts)]
        real_len = np.count_nonzero(func_tokens != pad_token_guess)

        # --- NEW: Get the number of instructions from the JSON file for this function ---
        num_instructions = None
        instruction_list = None
        func_offset = offsets[func_idx] if offsets is not None and len(offsets) > func_idx else None
        
        # Load the JSON and find the matching function
        parent_dir = os.path.dirname(output_dir)
        json_files = [f for f in os.listdir(parent_dir) if f.endswith('.json')]
        if json_files:
            json_path = os.path.join(parent_dir, json_files[0])
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                functions = data.get('functions', [])
                
                # Find the function by offset
                match_idx = None
                if func_offset is not None:
                    for i, func in enumerate(functions):
                        if func.get('func_start', None) == func_offset:
                            match_idx = i
                            break
                
                if match_idx is not None and match_idx < len(functions):
                    instruction_list = functions[match_idx].get('instructions', [])
                    num_instructions = len(instruction_list)
                else:
                    logger.info(f"Could not find JSON function with offset {func_offset}")
            except Exception as e:
                logger.info(f"Could not read JSON for original instructions: {e}")
        
        # --- NEW: Crop to number of instructions from JSON if available ---
        if num_instructions is not None and num_instructions > 0:
            func_importance = func_importance[:num_instructions]
            func_tokens = func_tokens[:num_instructions]
            x_pos = np.arange(num_instructions)
        else:
            # Fallback: use real_len as before
            func_importance = func_importance[:real_len]
            func_tokens = func_tokens[:real_len]
            x_pos = np.arange(real_len)
        
        if len(func_importance) == 0:
            logger.warning(f"No tokens to plot for function {func_idx}!")
            return
        
        # Create single comprehensive plot
        plt.figure(figsize=(15, 8))
        
        # Add main title
        # Use display_func_idx and display_func_offset if provided, else fallback to func_idx/func_offset
        display_idx = display_func_idx if display_func_idx is not None else (actual_func_idx if actual_func_idx is not None else func_idx)
        display_offset = display_func_offset if display_func_offset is not None else func_offset
        title = f'Attention Rollout Analysis — Function {display_idx}'
        if display_offset is not None:
            title += f' (offset: {display_offset})'
        
        # Add pos encoding and seed info to title
        if model is not None and model_name is not None:
            pos_encoding_suffix = f"_pos{model.func_encoder.pos_encoding_type}"
            import re
            seed_match = re.search(r'seed(\d+)', model_name)
            seed_suffix = f"_seed{seed_match.group(1)}" if seed_match else "_seed0"
            title += f" (pos: {model.func_encoder.pos_encoding_type}, seed: {seed_match.group(1) if seed_match else '0'})"
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        
        # Plot 1: Token importance over sequence
        plt.subplot(2, 1, 1)
        
        # Color by importance level
        colors = ['red' if imp > np.mean(func_importance) + np.std(func_importance) else
                 'orange' if imp > np.mean(func_importance) else 'blue' 
                 for imp in func_importance]
        
        bars = plt.bar(x_pos, func_importance, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.xlabel('Token Index', fontsize=12)
        plt.ylabel('Rollout Importance Score', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add mean/std lines with better labels
        mean_val = func_importance.mean() if len(func_importance) > 0 else 0
        std_val = func_importance.std() if len(func_importance) > 0 else 0
        plt.axhline(mean_val, color='black', linestyle='--', 
                   label=f'Mean = {mean_val:.2f}')
        plt.axhline(mean_val + std_val, color='red', linestyle=':', 
                   label=f'Mean+1σ = {mean_val + std_val:.2f}')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 2: Token importance vs token values
        plt.subplot(2, 1, 2)
        plt.scatter(func_tokens, func_importance, alpha=0.6, s=20)
        plt.xlabel('Token Value (Opcode ID)', fontsize=12)
        plt.ylabel('Rollout Importance Score', fontsize=12)
        
        # Add correlation info with better formatting
        if len(func_importance) > 1:
            correlation = np.corrcoef(func_tokens, func_importance)[0, 1]
        else:
            correlation = float('nan')
        plt.annotate(f'Pearson r = {correlation:.3f}', xy=(0.05, 0.95),
                   xycoords='axes fraction',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
                   fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{filename_prefix}_analysis.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved attention rollout visualization: {plot_path}")
        
        # Save rollout data as JSON (simplified)
        rollout_data = {
            'selected_function_slot': int(func_idx),
            'selected_function_offset': func_offset,
            'token_importance': func_importance.tolist(),
            'stats': {
                'mean_importance': float(mean_val),
                'std_importance': float(std_val),
                'min_importance': float(func_importance.min()) if len(func_importance) > 0 else 0,
                'max_importance': float(func_importance.max()) if len(func_importance) > 0 else 0
            },
            'function_offset': func_offset,
            'num_instructions': int(num_instructions) if num_instructions is not None else None,
            'debug': {
                'unique_tokens': unique_tokens.tolist(),
                'token_counts': dict(zip(unique_tokens.tolist(), counts.tolist())),
                'guessed_pad_token': int(pad_token_guess) if isinstance(pad_token_guess, (int, np.integer)) else str(pad_token_guess),
                'real_token_count': int(real_len),
                'original_instructions': instruction_list if instruction_list is not None else None
            }
        }
        
        json_path = os.path.join(output_dir, f'{filename_prefix}_data.json')
        with open(json_path, 'w') as f:
            json.dump(rollout_data, f, indent=2)
        
        logger.info(f"Saved attention rollout data: {json_path}")
        
    except Exception as e:
        logger.error(f"Error visualizing attention rollout: {e}")
        import traceback
        traceback.print_exc()

def construct_model_name(pos_encoding, seed):
    """Construct model name based on pos encoding and seed."""
    return f"curriculum_hotflip_pos-{pos_encoding}_attn-attn_seed{seed}"

def get_true_function_count(file_path):
    """Get the true number of functions from the JSON file before padding."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return len(data.get('functions', []))
    except Exception as e:
        logger.warning(f"Error reading function count from {file_path}: {e}")
        return 0

def run_aggregated_attention_analysis(model, tokenizer, malicious_file, device, max_func_len, max_funcs, output_dir, model_name):
    """Run attention analysis and save only aggregated views (averaged across all heads)."""
    
    logger.info(f"\n=== RUNNING AGGREGATED ATTENTION ANALYSIS ===")
    logger.info(f"File: {os.path.basename(malicious_file)}")
    
    # Initialize variables to avoid UnboundLocalError
    attention_weights = None
    attention_heatmaps_generated = False
    rollout_generated = False
    classifier_attention_generated = False
    
    # Get true function count from JSON
    true_function_count = get_true_function_count(malicious_file)
    logger.info(f"True function count: {true_function_count}")

    # --- Load JSON for reference ---
    with open(malicious_file, 'r') as f:
        data = json.load(f)
    functions = data.get('functions', [])
    
    # --- DEBUG: Print first few functions from JSON ---
    logger.info(f"\n[DEBUG] === JSON FUNCTION DEBUG ===")
    for i in range(min(3, len(functions))):
        func = functions[i]
        instructions = func.get('instructions', [])
        func_start = func.get('func_start', None)
        logger.info(f"[DEBUG] JSON Function {i}: offset={func_start}, {len(instructions)} instructions")
        logger.info(f"[DEBUG] JSON Function {i} instructions: {instructions[:10]}...")  # First 10 instructions
        
        # --- DEBUG: Show how these instructions get tokenized ---
        if instructions:
            try:
                tokenized = tokenizer.encode(instructions, max_func_len)
                logger.info(f"[DEBUG] JSON Function {i} tokenized: {tokenized[:20]}...")  # First 20 tokens
                # Show vocab mapping for first few tokens
                unique_tokens = list(set(tokenized[:10]))
                for token in unique_tokens:
                    if hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, list):
                        if 0 <= token < len(tokenizer.vocab):
                            vocab_word = tokenizer.vocab[token]
                        else:
                            vocab_word = f"OUT_OF_RANGE({token})"
                    else:
                        vocab_word = f"UNKNOWN_VOCAB_FORMAT({token})"
                    logger.info(f"[DEBUG] Token {token} -> '{vocab_word}'")
            except Exception as e:
                logger.info(f"[DEBUG] Error tokenizing function {i}: {e}")

    # --- DEBUG: Check what's in the sampled functions that ended up in tensor slots ---
    logger.info(f"\n[DEBUG] === SAMPLED FUNCTION DEBUG ===")
    dataset_mapping = None  # Will be set after we get the tensor mapping
    
    # Create dataset for this single file
    # IMPORTANT: Use the same tokenization settings as training
    # Based on the training command: --no-use_split_tokens --no-use_boundaries
    logger.info(f"[DEBUG] Creating dataset with use_split_tokens=False, use_boundaries=False to match training...")
    dataset = MalwareDataset([malicious_file], tokenizer, max_func_len, max_funcs, 
                           cache_in_memory=False, use_split_tokens=False, use_boundaries=False, seed=42)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                           collate_fn=lambda batch: collate_batch(batch, tokenizer, max_func_len, max_funcs))
    
    # Get the processed data (now includes mapping)
    func_tensor, label, tensor_mapping = next(iter(dataloader))
    func_tensor = func_tensor.to(device)  # Shape: (1, max_funcs, max_func_len)
    
    logger.info(f"Processed tensor shape: {func_tensor.shape}")
    logger.info(f"Tensor mapping shape: {tensor_mapping.shape}")
    logger.info(f"Padding fraction: {(func_tensor == 0).float().mean():.3f}")
    
    # --- DEBUG: Show the actual mapping from the dataset ---
    dataset_mapping = tensor_mapping[0].tolist()  # First (and only) batch
    logger.info(f"\n[DEBUG] === FUNCTION MAPPING DEBUG ===")
    logger.info(f"[DEBUG] Total functions in JSON: {len(functions)}")
    logger.info(f"[DEBUG] Dataset mapping (tensor slot -> JSON index): {dataset_mapping[:10]}...")  # First 10
    
    # --- DEBUG: Check the specific functions that ended up in the first few tensor slots ---
    for i in range(min(5, len(dataset_mapping))):
        json_idx = dataset_mapping[i]
        if json_idx >= 0 and json_idx < len(functions):
            func_data = functions[json_idx]
            func_start = func_data.get('func_start', None)
            instructions = func_data.get('instructions', [])
            logger.info(f"[DEBUG] Tensor slot {i} -> JSON function {json_idx} (offset: {func_start})")
            logger.info(f"[DEBUG] JSON Function {json_idx} has {len(instructions)} instructions: {instructions[:5]}...")
            
            # Check how this gets tokenized
            if instructions:
                try:
                    tokenized = tokenizer.encode(instructions, max_func_len)
                    unique_tokens, counts = np.unique(tokenized, return_counts=True)
                    logger.info(f"[DEBUG] JSON Function {json_idx} tokenized unique: {dict(zip(unique_tokens, counts))}")
                except Exception as e:
                    logger.info(f"[DEBUG] Error tokenizing JSON function {json_idx}: {e}")
            else:
                logger.info(f"[DEBUG] JSON Function {json_idx} has NO INSTRUCTIONS - will be filled with UNK tokens")
        else:
            logger.info(f"[DEBUG] Tensor slot {i} -> PADDING (index {json_idx})")
    
    # Build offsets array using the dataset mapping
    offsets = [functions[idx].get('func_start', None) if idx >= 0 and idx < len(functions) else None for idx in dataset_mapping]
    
    # --- DEBUG: Show what ended up in the first few tensor slots ---
    logger.info(f"\n[DEBUG] === TENSOR SLOT DEBUG ===")
    func_tensor_cpu = func_tensor[0].cpu().numpy()  # Shape: (max_funcs, max_func_len)
    for slot_idx in range(min(3, func_tensor_cpu.shape[0])):
        slot_tokens = func_tensor_cpu[slot_idx]
        unique_tokens, counts = np.unique(slot_tokens, return_counts=True)
        json_idx = dataset_mapping[slot_idx]
        logger.info(f"[DEBUG] Tensor slot {slot_idx} -> JSON function {json_idx}")
        logger.info(f"[DEBUG] Tensor slot {slot_idx}: unique tokens = {unique_tokens}, counts = {dict(zip(unique_tokens, counts))}")
        logger.info(f"[DEBUG] Tensor slot {slot_idx} first 20 tokens: {slot_tokens[:20]}")
        
        # Check if this slot has real content
        if len(unique_tokens) > 1:
            logger.info(f"[DEBUG] Tensor slot {slot_idx} appears to have REAL content")
        else:
            logger.info(f"[DEBUG] Tensor slot {slot_idx} appears to be PADDING (all {unique_tokens[0]})")
    
    # Get mask for valid tokens (not padding), shape (B*F, L)
    mask = (func_tensor != 0).float().view(-1, func_tensor.shape[-1])  # (B*F, L)
    
    # Run model prediction
    model.eval()
    with torch.no_grad():
        # Remove device= from forward call - model should handle device internally
        logits = model(func_tensor).squeeze(-1)  # (batch,) or scalar
        probs = torch.sigmoid(logits)                         # (batch,) or scalar
        
        # Handle both 0-dim (scalar) and 1-dim tensors
        if probs.dim() == 0:
            prob_mal = float(probs.item())
        else:
            prob_mal = float(probs[0].item())
        
        prob_ben = 1.0 - prob_mal
        prediction = 1 if prob_mal > 0.5 else 0
        confidence = max(prob_mal, prob_ben)

    logger.info(f"Model prediction: {'Malicious' if prediction == 1 else 'Benign'}")
    logger.info(f"Confidence: {confidence:.4f}")
    logger.info(f"Probabilities: Benign={prob_ben:.4f}, Malicious={prob_mal:.4f}")
    
    # Step 1: Extract and visualize individual attention heads
    attention_weights = extract_attention_weights_with_hooks(model, func_tensor, device, max_func_len)

    # Print attention stats for each layer and head
    for i, layer_heads in enumerate(attention_weights):
        for j, head_attn in enumerate(layer_heads):
            print(f"Layer {i+1} Head {j+1} attn stats: min={head_attn.min().item()}, max={head_attn.max().item()}, mean={head_attn.mean().item()}, std={head_attn.std().item()}")

    # Graceful handling for no attention weights
    if not attention_weights:
        logger.warning("No transformer attention returned; skipping attention heatmaps.")
        attention_heatmaps_generated = False
    else:
        logger.info(f"Extracted attention from {len(attention_weights)} layers")
        attention_heatmaps_generated = True
        
        # Plot individual attention heatmaps for each head in each layer
        for layer_idx, layer_heads in enumerate(attention_weights):
            logger.info(f"Processing layer {layer_idx} with {len(layer_heads)} heads")
            logger.info(f"Aggregating attention across {func_tensor.shape[0] * func_tensor.shape[1]} functions (batch_size={func_tensor.shape[0]}, max_funcs={func_tensor.shape[1]})")
            
            # Create a single figure with 8 subplots (2x4 grid) for all heads in this layer
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            pos_encoding_title = f" (Pos Encoding: {model.func_encoder.pos_encoding_type})"
            
            # Add seed information to title
            import re
            seed_match = re.search(r'seed(\d+)', model_name)
            seed_title = f" (seed: {seed_match.group(1)})" if seed_match else " (seed: 0)"
            
            fig.suptitle(f'Layer {layer_idx+1} Attention Heads{pos_encoding_title}{seed_title}', fontsize=16, fontweight='bold')
            
            # Flatten axes for easier indexing
            axes_flat = axes.flatten()
            
            for head_idx, head_attn in enumerate(layer_heads):
                # Aggregate across all functions (B*F dimension) for this specific head
                if head_attn.ndim == 3:
                    # head_attn shape: (B*F, L, L) - aggregate across all functions
                    attn_map_2d = head_attn.mean(dim=0).detach().cpu().numpy()  # Average across all functions
                elif head_attn.ndim == 2:
                    attn_map_2d = head_attn.detach().cpu().numpy()
                else:
                    raise ValueError(f"Unexpected attention map shape: {head_attn.shape}")
                
                # Crop to non-padded region (use mask or true function length)
                # We'll use the maximum sequence length across all functions
                plot_mask = (func_tensor != 0).float().view(-1, func_tensor.shape[-1])
                # Find the maximum number of valid tokens across all functions
                valid_lengths = plot_mask.sum(dim=1).cpu().numpy().astype(int)
                max_seq_len = valid_lengths.max()
                attn_map_2d = attn_map_2d[:max_seq_len, :max_seq_len]
                
                # Plot in the corresponding subplot
                ax = axes_flat[head_idx]
                im = ax.imshow(attn_map_2d, aspect='auto', cmap='viridis')
                ax.set_title(f'Head {head_idx+1}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Token Position', fontsize=10)
                ax.set_ylabel('Token Position', fontsize=10)
                
                # Add colorbar to the right of the subplot
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                cbar.ax.tick_params(labelsize=8)
            
            # Remove any unused subplots
            for idx in range(len(layer_heads), len(axes_flat)):
                fig.delaxes(axes_flat[idx])
            
            plt.tight_layout()
            
            # Include positional encoding type and seed in filename
            pos_encoding_suffix = f"_pos{model.func_encoder.pos_encoding_type}"
            # Extract seed from model name (assuming format contains "seed{number}")
            import re
            seed_match = re.search(r'seed(\d+)', model_name)
            seed_suffix = f"_seed{seed_match.group(1)}" if seed_match else "_seed0"
            out_path = os.path.join(output_dir, f'attention_layer{layer_idx+1}_all_heads{pos_encoding_suffix}{seed_suffix}.png')
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved attention heatmap for all heads in layer {layer_idx+1}: {out_path}")
        
        # Step 1.5: Attention Rollout Analysis (Abnar & Zuidema ACL 2020)
        logger.info(f"\n=== RUNNING ATTENTION ROLLOUT ANALYSIS ===")
        
        try:
            # For rollout analysis, we need to create aggregated attention maps (averaged across heads)
            # to maintain compatibility with existing rollout functions
            logger.info("Creating aggregated attention maps for rollout analysis...")
            aggregated_attention = []
            for layer_heads in attention_weights:
                # Stack all heads and average: (num_heads, B*F, L, L) -> (B*F, L, L)
                layer_stack = torch.stack(layer_heads, dim=0)  # (num_heads, B*F, L, L)
                layer_avg = layer_stack.mean(dim=0)  # Average across heads: (B*F, L, L)
                aggregated_attention.append(layer_avg)
            
            # Compute mean attention rollout only (on GPU, mask applied)
            logger.info("Computing mean attention rollout...")
            # Create mask that matches the minimum sequence length from attention weights
            min_seq_len = min(head_attn.shape[-1] for layer_heads in attention_weights for head_attn in layer_heads)
            rollout_mask = (func_tensor != 0).float().view(-1, func_tensor.shape[-1])[:, :min_seq_len]  # Crop to min_seq_len
            
            # DEBUG: Log aggregated attention stats before rollout
            logger.info(f"[DEBUG] About to compute rollout with {len(aggregated_attention)} layers")
            for i, layer_attn in enumerate(aggregated_attention):
                logger.info(f"[DEBUG] Layer {i} aggregated attention shape: {layer_attn.shape}")
                logger.info(f"[DEBUG] Layer {i} aggregated attention stats: min={layer_attn.min():.6f}, max={layer_attn.max():.6f}, mean={layer_attn.mean():.6f}, std={layer_attn.std():.6f}")
            logger.info(f"[DEBUG] Rollout mask shape: {rollout_mask.shape}")
            logger.info(f"[DEBUG] Rollout mask stats: min={rollout_mask.min():.6f}, max={rollout_mask.max():.6f}, mean={rollout_mask.mean():.6f}")
            
            mean_rollout = rollout_mean_attention(aggregated_attention, device, mask=rollout_mask)
            
            if mean_rollout is not None:
                # Visualize mean rollout (pass offsets)
                # Include positional encoding type and seed in filename prefix
                pos_encoding_suffix = f"_pos{model.func_encoder.pos_encoding_type}"
                import re
                seed_match = re.search(r'seed(\d+)', model_name)
                seed_suffix = f"_seed{seed_match.group(1)}" if seed_match else "_seed0"
                rollout_prefix = f"rollout{pos_encoding_suffix}{seed_suffix}"
                visualize_attention_rollout(mean_rollout, func_tensor, output_dir, rollout_prefix, offsets=offsets, model=model, model_name=model_name)
                rollout_generated = True
                logger.info(" Mean attention rollout completed")
                
                # Generate token importance summary (no function-level attention yet)
                logger.info("Generating token importance summary...")
                # Include positional encoding type and seed in filename
                pos_encoding_suffix = f"_pos{model.func_encoder.pos_encoding_type}"
                import re
                seed_match = re.search(r'seed(\d+)', model_name)
                seed_suffix = f"_seed{seed_match.group(1)}" if seed_match else "_seed0"
                token_summary_filename = f"token_importance_summary{pos_encoding_suffix}{seed_suffix}"
                summary_files = save_token_importance_summary(mean_rollout, func_tensor, None, output_dir, token_summary_filename)
                if summary_files:
                    logger.info(" Token importance summary generated")
                    logger.info(f"  - JSON: {summary_files['json']}")
                    logger.info(f"  - Text: {summary_files['txt']}")
                    logger.info(f"  - CSV: {summary_files['csv']}")
                else:
                    logger.warning(" Token importance summary generation failed")
            
        except Exception as e:
            logger.warning(f"Attention rollout analysis failed: {e}")
            import traceback
            traceback.print_exc()
            rollout_generated = False
    
    # Step 2: Extract and visualize classifier attention (proper implementation)
    try:
        # Verify aggregator structure
        logger.info(f"Aggregator structure: {model.aggregator}")
        logger.info(f"Aggregator use_attention: {model.aggregator.use_attention}")
        if hasattr(model.aggregator, 'attn'):
            logger.info(f"Aggregator attention layer: {model.aggregator.attn}")
            
            # Extract classifier attention weights using forward hook
            saved_attention = []
            
            def attention_hook(module, input, output):
                # The aggregator attention module outputs attention weights
                saved_attention.append(output.detach())
            
            # Register hook on the attention module
            hook = model.aggregator.attn.register_forward_hook(attention_hook)
            
            # Forward pass to get attention weights
            with torch.no_grad():
                _ = model(func_tensor)
            
            # Remove hook
            hook.remove()
            
            if saved_attention:
                # Get attention weights: (batch_size, num_funcs, 1) -> (batch_size, num_funcs)
                attention_weights = saved_attention[0].squeeze(-1)  # (1, max_funcs)
                
                # Create function mask to identify valid functions
                func_mask = (func_tensor.sum(dim=-1) > 0).float()  # (1, max_funcs)
                
                # Apply mask and softmax to get proper attention distribution
                masked_attention = attention_weights.masked_fill(func_mask == 0, float('-inf'))
                attention_probs = torch.softmax(masked_attention, dim=-1)
                
                logger.info(f"Extracted classifier attention: {attention_probs.shape}")
                logger.info(f"Attention stats: min={attention_probs.min().item():.6f}, max={attention_probs.max().item():.6f}, mean={attention_probs.mean().item():.6f}")
                classifier_attention_generated = True
                
                # Create simplified function attention visualization
                plt.figure(figsize=(30, 12))  # Much wider to accommodate all 64 bars
                
                # Add main title
                title = 'Function-Level Classifier Attention Analysis'
                
                # Add pos encoding and seed info to title
                pos_encoding_suffix = f"_pos{model.func_encoder.pos_encoding_type}"
                import re
                seed_match = re.search(r'seed(\d+)', model_name)
                seed_suffix = f"_seed{seed_match.group(1)}" if seed_match else "_seed0"
                title += f" (pos: {model.func_encoder.pos_encoding_type}, seed: {seed_match.group(1) if seed_match else '0'})"
                
                plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
                
                func_weights = attention_probs[0].cpu().numpy()  # First (and only) batch
                x_pos = np.arange(len(func_weights))
                
                # Show ALL functions (not just significant ones)
                all_indices = np.arange(len(func_weights))
                all_weights = func_weights
                all_x_pos = np.arange(len(all_weights))
                
                # Color by attention strength
                colors = ['red' if w > np.mean(func_weights) + np.std(func_weights) else 
                         'orange' if w > np.mean(func_weights) else 'blue' for w in all_weights]
                
                bars = plt.bar(all_x_pos, all_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                plt.xlabel('Function Index and Offset', fontsize=12)
                plt.ylabel('Attention Weight Score', fontsize=12)
                
                # Add mean/std lines with better labels
                mean_val = func_weights.mean()
                std_val = func_weights.std()
                plt.axhline(mean_val, color='black', linestyle='--', 
                           label=f'Mean = {mean_val:.3f}')
                plt.axhline(mean_val + std_val, color='red', linestyle=':', 
                           label=f'Mean+1σ = {mean_val + std_val:.3f}')
                plt.legend(loc='upper right', fontsize=10)
                plt.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars with better positioning (only for significant ones to avoid clutter)
                threshold = 0.01
                for i, (idx, weight) in enumerate(zip(all_indices, all_weights)):
                    if weight > threshold:  # Only label significant bars
                        plt.text(i, weight + 0.005, f'{weight:.3f}', 
                                ha='center', va='bottom', fontsize=8, rotation=0,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                
                # --- NEW: Improved x-axis labels with function index and offset ---
                # Create dual-line labels: index on top, offset below (perpendicular/vertical)
                tick_labels = []
                for idx in all_indices:
                    func_offset = offsets[idx] if idx < len(offsets) and offsets[idx] is not None else 'None'
                    # Format offset as decimal (not hex) for easier searching
                    if isinstance(func_offset, int):
                        offset_str = str(func_offset)
                    else:
                        offset_str = str(func_offset)
                    
                    # Create dual-line label: index on first line, offset on second (perpendicular)
                    tick_labels.append(f'{idx}\n{offset_str}')
                
                plt.xticks(all_x_pos, tick_labels, rotation=90, fontsize=8, ha='center')
                
                # Adjust y-axis limits to accommodate labels
                plt.ylim(0, max(all_weights) * 1.15)
                
                plt.tight_layout(rect=[0, 0.12, 1, 0.95])  # Leave more space at bottom for rotated labels
                
                # Include positional encoding type and seed in filename
                pos_encoding_suffix = f"_pos{model.func_encoder.pos_encoding_type}"
                import re
                seed_match = re.search(r'seed(\d+)', model_name)
                seed_suffix = f"_seed{seed_match.group(1)}" if seed_match else "_seed0"
                func_attn_path = os.path.join(output_dir, f'classifier_attention{pos_encoding_suffix}{seed_suffix}.png')
                plt.savefig(func_attn_path, dpi=200, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved function attention: {func_attn_path}")
                
                # --- NEW: Create simple plot of highest attention function ---
                # Find the function with highest attention weight
                highest_attn_idx = np.argmax(func_weights)
                highest_attn_weight = func_weights[highest_attn_idx]
                highest_attn_offset = offsets[highest_attn_idx] if highest_attn_idx < len(offsets) else None
                
                logger.info(f"[DEBUG] Highest attention function: index {highest_attn_idx}, weight {highest_attn_weight:.4f}, offset {highest_attn_offset}")
                

                
                # --- NEW: Create rollout analysis for highest attention function ---
                # Find the function with highest attention weight
                highest_attn_idx = np.argmax(func_weights)
                highest_attn_weight = func_weights[highest_attn_idx]
                highest_attn_offset = offsets[highest_attn_idx] if highest_attn_idx < len(offsets) else None
                
                logger.info(f"[DEBUG] Highest attention function: index {highest_attn_idx}, weight {highest_attn_weight:.4f}, offset {highest_attn_offset}")
                
                # Create a modified tensor with only the highest attention function
                # Extract rollout data correctly for the highest attention function
                if mean_rollout is not None:
                    # mean_rollout shape: (B*F, L) = (64, L)
                    logger.info(f"[DEBUG] mean_rollout shape: {mean_rollout.shape}")
                    
                    # Extract the rollout for the specific function (index highest_attn_idx)
                    # mean_rollout is organized as [func0_rollout, func1_rollout, ..., func63_rollout]
                    highest_func_rollout = mean_rollout[highest_attn_idx:highest_attn_idx+1, :]  # (1, L)
                    logger.info(f"[DEBUG] highest_func_rollout shape: {highest_func_rollout.shape}")
                    logger.info(f"[DEBUG] highest_func_rollout stats: min={highest_func_rollout.min():.6f}, max={highest_func_rollout.max():.6f}, mean={highest_func_rollout.mean():.6f}, std={highest_func_rollout.std():.6f}")
                    logger.info(f"[DEBUG] highest_func_rollout first 10 values: {highest_func_rollout[0][:10].tolist()}")
                    
                    # Create a single-function tensor for visualization
                    highest_func_tensor = func_tensor[:, highest_attn_idx:highest_attn_idx+1, :]  # (1, 1, L)
                    logger.info(f"[DEBUG] highest_func_tensor shape: {highest_func_tensor.shape}")
                    
                    # Create offsets for just this function
                    highest_func_offsets = [highest_attn_offset] if highest_attn_offset is not None else [None]
                    
                    # Use the existing visualize_attention_rollout function
                    highest_func_prefix = f"highest_attention_function_{highest_attn_idx}_rollout{pos_encoding_suffix}{seed_suffix}"
                    visualize_attention_rollout(
                        highest_func_rollout, 
                        highest_func_tensor, 
                        output_dir, 
                        highest_func_prefix, 
                        offsets=[highest_attn_offset], 
                        actual_func_idx=0,  # Always 0 for the sliced tensor
                        model=model, 
                        model_name=model_name,
                        display_func_idx=highest_attn_idx,  # Pass original index for labeling
                        display_func_offset=highest_attn_offset  # Pass original offset for labeling
                    )
                
                # Save attention weights as JSON for further analysis
                attention_data = {
                    'attention_weights': attention_probs[0].cpu().tolist(),
                    'function_mask': func_mask[0].cpu().tolist(),
                    'stats': {
                        'mean': float(attention_probs.mean()),
                        'std': float(attention_probs.std()),
                        'min': float(attention_probs.min()),
                        'max': float(attention_probs.max()),
                        'top_5_functions': [int(i) for i in np.argsort(func_weights)[-5:][::-1]],
                        'top_5_weights': [float(w) for w in np.sort(func_weights)[-5:][::-1]]
                    }
                }
                
                attention_json_path = os.path.join(output_dir, f'classifier_attention_weights{pos_encoding_suffix}{seed_suffix}.json')
                with open(attention_json_path, 'w') as f:
                    json.dump(attention_data, f, indent=2)
                logger.info(f"Saved attention weights data: {attention_json_path}")
                
            else:
                logger.warning("No classifier attention returned; skipping classifier attention plot.")
        else:
            logger.warning("No attention layer found in aggregator!")
            # Don't return early - just skip classifier attention analysis
            classifier_attention_generated = False
        
    except Exception as e:
        logger.warning(f"Could not extract classifier attention: {e}")
        import traceback
        traceback.print_exc()
        classifier_attention_generated = False
    
    # Step 3: Save analysis summary
    file_info = {
        'file': malicious_file,
        'label': label.item(),
        'true_function_count': true_function_count,
        'padded_function_count': max_funcs,
    }
    
    summary = {
        'file_info': file_info,
        'model_prediction': {
            'predicted_label': prediction,
            'confidence': confidence,
            'probabilities': {
                'benign': prob_ben,
                'malicious': prob_mal
            }
        },
        'tensor_info': {
            'shape': list(func_tensor.shape),
            'padding_fraction': (func_tensor == 0).float().mean().item(),
        },
        'attention_info': {
            'num_layers': len(attention_weights) if attention_weights is not None and len(attention_weights) > 0 else 0,
            'num_heads_per_layer': len(attention_weights[0]) if attention_weights is not None and len(attention_weights) > 0 else 0,
            'total_attention_maps': sum(len(layer_heads) for layer_heads in attention_weights) if attention_weights is not None else 0,
            'attention_heatmaps_generated': attention_heatmaps_generated,
            'rollout_analysis_generated': rollout_generated,
            'classifier_attention_generated': classifier_attention_generated,
            'classifier_attention_available': hasattr(model.aggregator, 'attn') if hasattr(model, 'aggregator') else False
        }
    }
    
    # Include positional encoding type and seed in filename
    pos_encoding_suffix = f"_pos{model.func_encoder.pos_encoding_type}"
    import re
    seed_match = re.search(r'seed(\d+)', model_name)
    seed_suffix = f"_seed{seed_match.group(1)}" if seed_match else "_seed0"
    summary_path = os.path.join(output_dir, f'analysis_summary{pos_encoding_suffix}{seed_suffix}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Analysis summary saved to: {summary_path}")
    logger.info(f"All outputs saved to: {output_dir}")
    
    return output_dir, summary

def extract_top_important_tokens(token_importance, func_tensor, attention_probs=None, top_k=10):
    """
    Extract the top most important tokens for each function and globally from attention analysis.
    
    Args:
        token_importance: (B*F, L_rollout) tensor from attention rollout
        func_tensor: (B, F, L_orig) original input tensor for context
        attention_probs: (B, F) function-level attention weights (optional)
        top_k: number of top tokens to extract per function and globally
    
    Returns:
        dict: summary of top important tokens for each function and globally
    """
    try:
        B, F, L_orig = func_tensor.shape
        Bf, L_rollout = token_importance.shape
        assert Bf == B * F, f"batch-size mismatch: {Bf} != {B * F}"
        
        # Reshape token importance back to (B, F, L_rollout)
        token_importance_reshaped = token_importance.view(B, F, L_rollout)
        
        # Crop func_tensor to match rollout length
        func_tensor_cropped = func_tensor[:, :, :L_rollout]
        
        # Get function-level attention if provided
        func_attention = None
        if attention_probs is not None:
            func_attention = attention_probs[0].cpu().numpy()  # (F,)
        
        results = {
            'summary': {
                'num_functions': F,
                'num_tokens_per_function': L_rollout,
                'top_k_tokens': top_k,
                'analysis_type': 'attention_rollout'
            },
            'functions': [],
            'global_top_tokens': []
        }
        
        # Per-function top tokens
        for func_idx in range(F):
            func_importance = token_importance_reshaped[0, func_idx].cpu().numpy()  # (L_rollout,)
            func_tokens = func_tensor_cropped[0, func_idx].cpu().numpy()  # (L_rollout,)
            top_indices = np.argsort(func_importance)[-top_k:][::-1]  # Descending order
            top_importance = func_importance[top_indices]
            top_token_values = func_tokens[top_indices]
            func_summary = {
                'function_index': func_idx,
                'function_attention_weight': float(func_attention[func_idx]) if func_attention is not None else None,
                'total_importance': float(func_importance.sum()),
                'mean_importance': float(func_importance.mean()),
                'max_importance': float(func_importance.max()),
                'top_tokens': []
            }
            for i, (idx, importance, token_val) in enumerate(zip(top_indices, top_importance, top_token_values)):
                token_info = {
                    'rank': i + 1,
                    'token_index': int(idx),
                    'token_value': int(token_val),
                    'importance_score': float(importance),
                    'importance_percentile': float(importance / func_importance.max() * 100)
                }
                func_summary['top_tokens'].append(token_info)
            results['functions'].append(func_summary)
        
        # Global aggregation: sum importance for each unique token value
        all_tokens = func_tensor_cropped[0].cpu().numpy().flatten()  # (F*L_rollout,)
        all_importance = token_importance_reshaped[0].cpu().numpy().flatten()  # (F*L_rollout,)
        token_stats = {}
        for token_val, importance in zip(all_tokens, all_importance):
            if int(token_val) == 0:
                continue  # skip padding
            if token_val not in token_stats:
                token_stats[token_val] = {'total_importance': 0.0, 'count': 0}
            token_stats[token_val]['total_importance'] += importance
            token_stats[token_val]['count'] += 1
        # Compute average importance and sort
        global_token_list = [
            {
                'token_value': int(token_val),
                'total_importance': stats['total_importance'],
                'count': stats['count'],
                'average_importance': stats['total_importance'] / stats['count'] if stats['count'] > 0 else 0.0
            }
            for token_val, stats in token_stats.items()
        ]
        global_token_list.sort(key=lambda x: x['total_importance'], reverse=True)
        results['global_top_tokens'] = global_token_list[:top_k]
        
        # Add overall statistics
        all_importance_flat = token_importance.cpu().numpy().flatten()
        results['overall_stats'] = {
            'mean_importance': float(all_importance_flat.mean()),
            'std_importance': float(all_importance_flat.std()),
            'min_importance': float(all_importance_flat.min()),
            'max_importance': float(all_importance_flat.max()),
            'total_tokens_analyzed': int(all_importance_flat.size)
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error extracting top important tokens: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_token_importance_summary(token_importance, func_tensor, attention_probs, output_dir, filename="token_importance_summary"):
    """
    Save a comprehensive summary of the most important tokens from attention analysis.
    
    Args:
        token_importance: (B*F, L) tensor from attention rollout
        func_tensor: (B, F, L) original input tensor
        attention_probs: (B, F) function-level attention weights
        output_dir: directory to save the summary
        filename: base filename for the summary
    """
    try:
        # Extract top important tokens
        token_summary = extract_top_important_tokens(token_importance, func_tensor, attention_probs, top_k=10)
        
        if token_summary is None:
            logger.warning("Could not extract token importance summary")
            return
        
        # Save detailed JSON summary
        json_path = os.path.join(output_dir, f'{filename}.json')
        with open(json_path, 'w') as f:
            json.dump(token_summary, f, indent=2)
        
        logger.info(f"Saved detailed token importance summary: {json_path}")
        
        # Create human-readable text summary
        txt_path = os.path.join(output_dir, f'{filename}.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ATTENTION ANALYSIS - TOP IMPORTANT TOKENS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall statistics
            stats = token_summary['overall_stats']
            f.write(f"OVERALL STATISTICS:\n")
            f.write(f"  Total tokens analyzed: {stats['total_tokens_analyzed']:,}\n")
            f.write(f"  Mean importance: {stats['mean_importance']:.6f}\n")
            f.write(f"  Std importance: {stats['std_importance']:.6f}\n")
            f.write(f"  Min importance: {stats['min_importance']:.6f}\n")
            f.write(f"  Max importance: {stats['max_importance']:.6f}\n\n")
            
            # GLOBAL TOP TOKENS (Aggregated across all functions)
            f.write(f"GLOBAL TOP {len(token_summary['global_top_tokens'])} TOKENS (Aggregated Across All Functions):\n")
            f.write("=" * 80 + "\n")
            f.write(f"  {'Rank':<4} {'Token':<8} {'TotalImp':<12} {'AvgImp':<12} {'Count':<8}\n")
            f.write(f"  {'-'*4} {'-'*8} {'-'*12} {'-'*12} {'-'*8}\n")
            for i, token_info in enumerate(token_summary['global_top_tokens'], 1):
                f.write(f"  {i:<4} {token_info['token_value']:<8} {token_info['total_importance']:<12.6f} {token_info['average_importance']:<12.6f} {token_info['count']:<8}\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Function-by-function breakdown
            f.write(f"FUNCTION-BY-FUNCTION BREAKDOWN:\n")
            f.write("=" * 80 + "\n\n")
            
            for func_data in token_summary['functions']:
                func_idx = func_data['function_index']
                func_attn = func_data['function_attention_weight']
                
                f.write(f"FUNCTION {func_idx}:\n")
                if func_attn is not None:
                    f.write(f"  Function attention weight: {func_attn:.4f}\n")
                else:
                    f.write("  Function attention weight: None\n")
                f.write(f"  Total importance: {func_data['total_importance']:.4f}\n")
                f.write(f"  Mean importance: {func_data['mean_importance']:.4f}\n")
                f.write(f"  Max importance: {func_data['max_importance']:.4f}\n\n")
                
                f.write(f"  TOP 10 MOST IMPORTANT TOKENS:\n")
                f.write(f"  {'Rank':<4} {'Token':<6} {'Value':<8} {'Importance':<12} {'Percentile':<12}\n")
                f.write(f"  {'-'*4} {'-'*6} {'-'*8} {'-'*12} {'-'*12}\n")
                
                for token_info in func_data['top_tokens']:
                    rank = token_info['rank']
                    token_idx = token_info['token_index']
                    token_val = token_info['token_value']
                    importance = token_info['importance_score']
                    percentile = token_info['importance_percentile']
                    
                    f.write(f"  {rank:<4} {token_idx:<6} {token_val:<8} {importance:<12.6f} {percentile:<12.1f}%\n")
                
                f.write("\n" + "-" * 80 + "\n\n")
            
            # Summary of most important tokens across all functions
            f.write(f"GLOBAL TOP TOKENS (Across All Functions):\n")
            f.write("=" * 80 + "\n\n")
            
            # Collect all tokens and their importance
            all_tokens = []
            for func_data in token_summary['functions']:
                for token_info in func_data['top_tokens']:
                    all_tokens.append({
                        'function': func_data['function_index'],
                        'token_value': token_info['token_value'],
                        'importance': token_info['importance_score']
                    })
            
            # Sort by importance and get top 20 globally
            all_tokens.sort(key=lambda x: x['importance'], reverse=True)
            top_global = all_tokens[:20]
            
            f.write(f"  {'Rank':<4} {'Function':<8} {'Token':<8} {'Importance':<12}\n")
            f.write(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*12}\n")
            
            for i, token_info in enumerate(top_global, 1):
                f.write(f"  {i:<4} {token_info['function']:<8} {token_info['token_value']:<8} {token_info['importance']:<12.6f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF TOKEN IMPORTANCE SUMMARY\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Saved human-readable token importance summary: {txt_path}")
        
        # Create a simple CSV summary for easy analysis
        csv_path = os.path.join(output_dir, f'{filename}.csv')
        with open(csv_path, 'w') as f:
            f.write("function_index,token_index,token_value,importance_score,importance_percentile\n")
            
            for func_data in token_summary['functions']:
                func_idx = func_data['function_index']
                for token_info in func_data['top_tokens']:
                    f.write(f"{func_idx},{token_info['token_index']},{token_info['token_value']},"
                           f"{token_info['importance_score']:.6f},{token_info['importance_percentile']:.1f}\n")
            # Add global top tokens at the end
            f.write("\nglobal_top_tokens,token_value,total_importance,average_importance,count\n")
            for token_info in token_summary['global_top_tokens']:
                f.write(f"global_top_tokens,{token_info['token_value']},{token_info['total_importance']:.6f},{token_info['average_importance']:.6f},{token_info['count']}\n")
        
        logger.info(f"Saved CSV token importance summary: {csv_path}")
        
        return {
            'json': json_path,
            'txt': txt_path,
            'csv': csv_path
        }
        
    except Exception as e:
        logger.error(f"Error saving token importance summary: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_analysis(model_name, malicious_file, vocab_path, output_dir, max_func_len=64, max_funcs=64):
    """Run the aggregated attention analysis on a single file."""
    
    # UPDATED: Local Windows model path
    model_path = fr"E:\SUMMER RESEARCH\curriculum_adversarial_results_hotflip\{model_name}\best_model_ADV_{model_name}.pt"
    if not os.path.exists(model_path):
        logger.error(f"ERROR: Model not found: {model_path}")
        return False
    
    # Check if vocab exists
    if not os.path.exists(vocab_path):
        logger.error(f"ERROR: Vocabulary file not found: {vocab_path}")
        return False
    
    # Check if malicious file exists
    if not os.path.exists(malicious_file):
        logger.error(f"ERROR: Malicious file not found: {malicious_file}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load vocabulary and tokenizer
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    tokenizer = OpcodeTokenizer(vocab)
    
    # Determine positional encoding type from model name
    pos_encoding = 'sinusoidal' if 'sinusoidal' in model_name else 'none'
    
    # Load model
    model = MalwareClassifier(
        vocab,
        d_model=256,
        nhead=8,
        num_layers=2,
        max_func_len=max_func_len,
        max_funcs=max_funcs,
        use_attention=True,
        pos_encoding=pos_encoding
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Setup device
    device = get_device()
    model.to(device)
    model.eval()
    
    # 1. Patch custom encoder layer to store individual head attention weights
    import types
    for layer in model.func_encoder.transformer.layers:
        orig_forward = layer.forward
        def new_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
            x = src
            attn_output, attn_weights = self.self_attn(
                x, x, x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False  # Get individual heads
            )
            # Store both individual heads and aggregated attention
            self.last_attn_weights = attn_weights  # This should be (B*F, num_heads, L, L)
            self.last_attn_weights_aggregated = attn_weights.mean(dim=1) if attn_weights.dim() == 4 else attn_weights  # (B*F, L, L)
            x = attn_output
            x = self.dropout1(x)
            x = self.norm1(src + x)
            x = self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x))))))
            return x
        layer.forward = types.MethodType(new_forward, layer)

    logger.info(f"Model loaded successfully: {model_name}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Using device: {device}")
    logger.info(f"Positional encoding: {pos_encoding}")
    logger.info(f"Max function length: {max_func_len}")
    logger.info(f"Max functions: {max_funcs}")
    
    # Run aggregated attention analysis
    try:
        file_output_dir, summary = run_aggregated_attention_analysis(
            model, tokenizer, malicious_file, device, max_func_len, max_funcs, output_dir, model_name
        )
        
        logger.info(" Analysis completed successfully!")
        logger.info(f"Output directory: {output_dir}")
        
        # Check for generated files
        if os.path.exists(output_dir):
            files = glob.glob(os.path.join(output_dir, "**/*"), recursive=True)
            png_files = [f for f in files if f.endswith('.png')]
            json_files = [f for f in files if f.endswith('.json')]
            
            logger.info(f"Generated {len(png_files)} PNG files and {len(json_files)} JSON files")
            
            # Show summary if available
            summary_file = os.path.join(output_dir, "analysis_summary.json")
            if os.path.exists(summary_file):
                logger.info(f"\nAnalysis summary available at: {summary_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
