"""
Adversarial Training with HotFlip Attack for Malware Classifier
-------------------------------------------------------------
This script implements adversarial training using HotFlip attacks to improve model robustness.
It trains a malware classifier while periodically generating adversarial examples and
incorporating them into the training process.

Usage:
    python train_adversarial_hotflip.py --data_glob './data/*/disassembled/*.json' [other options]

Dependencies:
    - torch, numpy, tqdm
    - model.py, dataset.py, build_vocabulary.py
"""

import os
import glob
import random
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Model Files'))
from model import MalwareClassifier, get_device
from dataset import MalwareDataset, collate_batch
from tqdm import tqdm
import time
from functools import partial
import argparse
from build_vocabulary import build_vocab, load_vocab, save_vocab
import subprocess
import shutil

# Import hotflip_attack from Attack Files directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Attack Files'))
from hotflip_attack import hotflip_attack, get_hardcoded_attack_configs
import tempfile

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('adversarial_training_hotflip.log')
    ]
)
logger = logging.getLogger(__name__)

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def hotflip_attack_adversarial_training(model, tokenizer, func_tensor, label, k, debug=False):
    """Wrapper for HotFlip attack specifically for adversarial training"""
    return hotflip_attack(model, tokenizer, func_tensor, label, k, debug)

def generate_and_save_adversarial_examples_hotflip(
    model: MalwareClassifier, 
    train_loader: DataLoader, 
    args: argparse.Namespace, 
    device: torch.device
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Generate successful HotFlip adversarial examples for malicious samples and save them
    Returns a dictionary mapping batch indices to successful adversarial examples (like PGD script).
    """
    adv_examples_dir = os.path.join(args.results_dir, 'adversarial_examples_hotflip')
    os.makedirs(adv_examples_dir, exist_ok=True)

    attack_configs = get_hardcoded_attack_configs()
    logger.info(f"Using HotFlip mixed attack strengths: {len(attack_configs)} configurations")
    for i, config in enumerate(attack_configs):
        logger.info(f"  Config {i+1}: k_percent={config['k_percent']} ({config['name']})")
    logger.info("Mixing strategy: even distribution (round-robin)")

    model.eval()
    adversarial_examples = {}
    total_malicious = 0
    successful_attacks = 0
    skipped_empty = 0
    skipped_errors = 0
    config_stats = {config['name']: {'attempted': 0, 'successful': 0} for config in attack_configs}

    for batch_idx, (batch_funcs, batch_labels) in enumerate(tqdm(train_loader, desc="Generating successful HotFlip adversarial examples")):
        malicious_mask = (batch_labels == 1)
        batch_funcs = batch_funcs.to(device)
        batch_labels = batch_labels.to(device)
        if not malicious_mask.any():
            continue
        
        malicious_indices = torch.where(malicious_mask)[0].cpu().tolist()
        batch_successful_examples = []
        for sample_idx in malicious_indices:
            total_malicious += 1
            attack_config = attack_configs[(total_malicious - 1) % len(attack_configs)]
            config_stats[attack_config['name']]['attempted'] += 1
            single_func = batch_funcs[sample_idx:sample_idx+1]  # (1, F, L)
            label = 1
            
            # CRITICAL: Check if sample has any valid functions (at least one non-zero token)
            has_valid_functions = (single_func > 0).any()
            if not has_valid_functions:
                skipped_empty += 1
                continue
            
            # Get original prediction
            try:
                with torch.no_grad():
                    orig_logit = model(single_func, device=device)
                    orig_pred = (torch.sigmoid(orig_logit) > 0.5).float()
            except (RuntimeError, ValueError) as e:
                skipped_errors += 1
                continue
            
            if orig_pred.item() != 1.0:
                continue
            total_valid_tokens = (single_func > 0).sum().item()
            k = max(1, int(attack_config['k_percent'] * total_valid_tokens))
            try:
                adv_tensor, adv_logits, flipped_positions, attack_info = hotflip_attack_adversarial_training(
                    model, model.tokenizer, single_func, label, k, debug=False
                )
                with torch.no_grad():
                    adv_pred = (torch.sigmoid(adv_logits) > 0.5).float()
                if orig_pred.item() == 1.0 and adv_pred.item() == 0.0:
                    successful_attacks += 1
                    config_stats[attack_config['name']]['successful'] += 1
                    batch_successful_examples.append({
                        'adv_tokens': adv_tensor.cpu(),
                        'original_batch_idx': sample_idx,
                        'sample_idx': len(batch_successful_examples),
                        'attack_config': attack_config.copy()
                    })
            except ValueError as e:
                if "empty tensor" in str(e) or "no valid functions" in str(e):
                    skipped_empty += 1
                else:
                    skipped_errors += 1
                continue
            except Exception as e:
                skipped_errors += 1
                continue
        if len(batch_successful_examples) > 0:
            adversarial_examples[batch_idx] = {
                'adv_tokens': torch.cat([ex['adv_tokens'] for ex in batch_successful_examples], dim=0),
                'malicious_indices': malicious_indices,
                'successful_indices': list(range(len(batch_successful_examples))),
                'num_successful': len(batch_successful_examples)
            }
    success_rate = successful_attacks / total_malicious if total_malicious > 0 else 0.0
    logger.info("Attack Success Analysis:")
    logger.info(f"  Total malicious samples: {total_malicious}")
    logger.info(f"  Successful attacks: {successful_attacks}")
    logger.info(f"  Success rate: {success_rate*100:.1f}%")
    logger.info(f"  Batches with successful attacks: {len(adversarial_examples)}")
    if skipped_empty > 0:
        logger.info(f"  Skipped empty samples: {skipped_empty}")
    if skipped_errors > 0:
        logger.info(f"  Skipped due to errors: {skipped_errors}")
    logger.info("Per-configuration success rates (expected: gentle < medium < aggressive):")
    for config_name, stats in config_stats.items():
        attempted = stats['attempted']
        successful = stats['successful']
        config_rate = successful / attempted if attempted > 0 else 0.0
        logger.info(f"  {config_name}: {successful}/{attempted} ({config_rate*100:.1f}%)")
    
    # Save all adversarial examples in a single file (like PGD script)
    if len(adversarial_examples) > 0:
        adv_examples_path = os.path.join(adv_examples_dir, 'adversarial_examples.pt')
        torch.save(adversarial_examples, adv_examples_path)
        logger.info(f"Saved {len(adversarial_examples)} adversarial examples to {adv_examples_path}")
    
    return adversarial_examples

def collate_fn(batch, tokenizer, max_func_len=64, max_funcs=64):
    return collate_batch(batch, tokenizer, max_func_len, max_funcs)

def main(args: argparse.Namespace) -> None:
    set_all_seeds(args.seed)
    logger.info(f"Set all random seeds to {args.seed}")
    ablation_name = args.ablation_name or f"pos-{args.pos_encoding}_attn-{'attn' if args.use_attention else 'mean'}"
    logger.info("=== TRAINING CONFIG (HotFlip) ===")
    logger.info(f"Positional Encoding: {args.pos_encoding}, Aggregator: {'Attention' if args.use_attention else 'Mean'}")
    logger.info(f"Ablation Name: {ablation_name}")
    logger.info(f"Training Schedule: {args.warmup_epochs} warmup + {args.max_epochs - args.warmup_epochs} adversarial epochs")
    logger.info(f"Adversarial λ scaling: {args.adv_lambda_start} → {args.adv_lambda_end} (linear)")
    logger.info(f"Adversarial example regeneration: Epoch {args.warmup_epochs + 1} + every 5 adversarial epochs")
    device = get_device()
    logger.info(f"Using device: {device}")
    all_files = sorted(glob.glob(args.data_glob))
    logger.info(f"Found {len(all_files)} files before filtering.")
    # Filter out no-function files BEFORE splitting
    no_func_path = os.path.join(os.path.dirname(__file__), 'no_functions_files.txt')
    if os.path.exists(no_func_path):
        with open(no_func_path, 'r') as nf:
            no_func_files = set(line.strip().replace('\\', '/') for line in nf if line.strip())
        def filter_no_func(files):
            return [f for f in files if f.replace('\\', '/') not in no_func_files]
        all_files = filter_no_func(all_files)
        logger.info(f"After filtering: {len(all_files)} files remain.")
    else:
        logger.warning(f"Warning: {no_func_path} not found. No files filtered.")
    
    # Use the same split_dataset function as PGD script for proper shuffling
    def split_dataset(all_files, train_ratio=0.6, val_ratio=0.2, seed=42):
        random.seed(seed)
        random.shuffle(all_files)
        n = len(all_files)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train+n_val]
        test_files = all_files[n_train+n_val:]
        return train_files, val_files, test_files
    
    train_files, val_files, test_files = split_dataset(all_files, seed=args.seed)
    # Assert splits are disjoint
    assert set(train_files).isdisjoint(test_files), "Train/Test overlap!"
    assert set(val_files).isdisjoint(test_files),   "Val/Test overlap!"
    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    if args.vocab_path:
        opcode_vocab = load_vocab(args.vocab_path)
        logger.info(f"Using provided vocab file: {args.vocab_path}")
    else:
        vocab_path = os.path.join(args.results_dir, f'vocab_{ablation_name}.json')
        logger.info(f"Building vocab at: {vocab_path}")
        opcode_vocab = build_vocab(
            train_files, 
            min_freq=args.min_freq, 
            use_split_tokens=args.use_split_tokens,
            use_boundaries=args.use_boundaries
        )
        save_vocab(opcode_vocab, vocab_path)
    logger.info(f"Vocabulary size: {len(opcode_vocab)}")
    from model import OpcodeTokenizer
    tokenizer = OpcodeTokenizer(opcode_vocab)
    logger.info(f"max_func_len={args.max_func_len}, max_funcs={args.max_funcs}, batch_size={args.batch_size}")
    logger.info(f"Tokenization style: {'atomic/split' if args.use_split_tokens else 'merged/original'}")
    # Warn if vocab and tokenization style are mismatched
    if ('atomic' in args.vocab_path and not args.use_split_tokens) or ('merged' in args.vocab_path and args.use_split_tokens):
        logger.warning("Warning: Vocab file and tokenization style may be mismatched! Check your settings.")
    
    train_dataset = MalwareDataset(train_files, tokenizer, args.max_func_len, args.max_funcs, cache_in_memory=False, 
                                  use_split_tokens=args.use_split_tokens, use_boundaries=args.use_boundaries, seed=args.seed)
    val_dataset = MalwareDataset(val_files, tokenizer, args.max_func_len, args.max_funcs, cache_in_memory=False,
                                use_split_tokens=args.use_split_tokens, use_boundaries=args.use_boundaries, seed=args.seed)
    collate = partial(collate_fn, tokenizer=tokenizer, max_func_len=args.max_func_len, max_funcs=args.max_funcs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, 
                             num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, 
                           num_workers=16, pin_memory=True, persistent_workers=True)
    logger.info(f"Initializing model on device: {device}")
    model = MalwareClassifier(opcode_vocab, d_model=256, nhead=8, num_layers=2, 
                             max_func_len=args.max_func_len, max_funcs=args.max_funcs, dropout=0.2, 
                             use_attention=args.use_attention, pos_encoding=args.pos_encoding).to(device)
    model.tokenizer = tokenizer  # For HotFlip
    
    # Verification: Check alignment between tokenizer and embedding
    logger.info(f"→ tokenizer.vocab_size()     = {tokenizer.vocab_size()}")
    logger.info(f"→ embedding.num_embeddings   = {model.func_encoder.embedding.num_embeddings}")
    if tokenizer.vocab_size() == model.func_encoder.embedding.num_embeddings:
        logger.info(" Tokenizer and embedding dimensions are aligned!")
    else:
        logger.error(" ERROR: Tokenizer and embedding dimension mismatch!")
    
    # Print a sample batch for debugging
    logger.info("\nPrinting a sample batch from train_loader:")
    sample_batch = next(iter(train_loader))
    sample_funcs, sample_labels = sample_batch
    logger.info(f"Sample batch_funcs shape: {sample_funcs.shape}")
    logger.info(f"Sample batch_labels: {sample_labels}")
    logger.info(f"First function tokens in batch[0]: {sample_funcs[0,0,:10]}")
    logger.info(f"First label in batch: {sample_labels[0]}")
    
    # Additional diagnostics
    logger.info(f"Label distribution in batch: {sample_labels.unique(return_counts=True)}")
    logger.info(f"Total unique values in first function: {sample_funcs[0,0].unique().shape[0]}")
    logger.info(f"Label dtype: {sample_labels.dtype}")
    
    # CRITICAL: Check for out-of-bounds indices against actual embedding size
    max_token_id = sample_funcs.max().item()
    emb_slots = model.func_encoder.embedding.num_embeddings
    logger.info(f"Max token ID in batch: {max_token_id}")
    logger.info(f"Embedding slots available: {emb_slots}")
    
    if max_token_id >= emb_slots:
        raise RuntimeError(f"Token ID {max_token_id} >= embedding rows {emb_slots}. This will cause CUDA assertion errors!")
    else:
        logger.info(f" All token IDs are within embedding bounds (0-{emb_slots-1}).")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.scheduler_patience)
    pos_weight = torch.tensor([1.0], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # COMPREHENSIVE LOSS DEBUGGING
    logger.info("\n===== LOSS DEBUGGING =====")
    model.eval()
    with torch.no_grad():
        sample_labels_gpu = sample_labels.to(device)
        sample_logits = model(sample_funcs, device=device)
        
        logger.info(f"Logits shape: {sample_logits.shape}")
        logger.info(f"Logits dtype: {sample_logits.dtype}")
        logger.info(f"Logits device: {sample_logits.device}")
        logger.info(f"Labels shape: {sample_labels_gpu.shape}")
        logger.info(f"Labels dtype: {sample_labels_gpu.dtype}")
        logger.info(f"Labels device: {sample_labels_gpu.device}")
        
        logger.info(f"Raw logits (first 10): {sample_logits[:10]}")
        logger.info(f"Logits range: min={sample_logits.min():.4f}, max={sample_logits.max():.4f}, mean={sample_logits.mean():.4f}")
        
        # Convert logits to probabilities for inspection (NOT for loss calculation)
        sample_probs = torch.sigmoid(sample_logits)
        logger.info(f"Probabilities (first 10): {sample_probs[:10]}")
        logger.info(f"Prob range: min={sample_probs.min():.4f}, max={sample_probs.max():.4f}, mean={sample_probs.mean():.4f}")
        
        # Calculate loss manually and with criterion
        manual_loss = torch.nn.functional.binary_cross_entropy_with_logits(sample_logits, sample_labels_gpu)
        criterion_loss = criterion(sample_logits, sample_labels_gpu)
        
        logger.info(f"Manual BCEWithLogitsLoss: {manual_loss.item():.6f}")
        logger.info(f"Criterion loss: {criterion_loss.item():.6f}")
        
        # Check predictions
        predictions = (sample_probs > 0.5).float()
        accuracy = (predictions == sample_labels_gpu).float().mean()
        logger.info(f"Batch accuracy: {accuracy.item():.4f}")
        logger.info(f"Positive predictions: {predictions.sum().item()}/{len(predictions)}")
        
        # Check for potential issues
        if sample_logits.std() < 0.1:
            logger.warning(f"Warning: Logits have very low variance! Model might not be learning.")
        if (sample_labels_gpu == 0).all():
            logger.warning(f"Warning: All labels are 0! This explains low loss if model predicts negative logits.")
        if (sample_labels_gpu == 1).all():
            logger.warning(f"Warning: All labels are 1! This explains low loss if model predicts positive logits.")
    
    logger.info("===== END LOSS DEBUGGING =====\n")
    model.train()
    
    checkpoint_path = os.path.join(args.results_dir, f'checkpoint_{ablation_name}.pt')
    best_model_path_warmup = os.path.join(args.results_dir, f'best_model_WARMUP_{ablation_name}.pt')
    best_model_path_adv = os.path.join(args.results_dir, f'best_model_ADV_{ablation_name}.pt')
    best_model_path = os.path.join(args.results_dir, f'best_model_OVERALL_{ablation_name}.pt')
    # Initialize best losses and epochs
    best_val_loss = float('inf')
    best_val_loss_warmup = float('inf')
    best_val_loss_adv = float('inf')
    best_epoch_overall = None
    best_epoch_warmup = None
    best_epoch_adv = None
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # JSON trainlog setup (like PGD script)
    log_path = os.path.join(args.results_dir, f'trainlog_{ablation_name}.json')
    log_data = []
    
    # Checkpoint resumption logic
    start_epoch = 1
    if args.restart:
        logger.info("--restart flag set: ignoring existing checkpoints, starting from scratch")
    elif os.path.exists(checkpoint_path):
        logger.info(f"Found existing checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        best_val_loss_warmup = checkpoint.get('best_val_loss_warmup', best_val_loss_warmup)
        best_val_loss_adv = checkpoint.get('best_val_loss_adv', best_val_loss_adv)
        best_epoch_overall = checkpoint.get('best_epoch_overall', best_epoch_overall)
        best_epoch_warmup = checkpoint.get('best_epoch_warmup', best_epoch_warmup)
        best_epoch_adv = checkpoint.get('best_epoch_adv', best_epoch_adv)
        logger.info(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        logger.info(f"  best_val_loss_warmup={best_val_loss_warmup:.4f} (epoch {best_epoch_warmup})")
        logger.info(f"  best_val_loss_adv={best_val_loss_adv:.4f} (epoch {best_epoch_adv})")
        logger.info(f"  best_epoch_overall={best_epoch_overall}")
    else:
        logger.info("No existing checkpoint found, starting from epoch 1")
    
    # Phase 1: Warmup training (clean data only)
    if start_epoch <= args.warmup_epochs:
        logger.info(f"=== PHASE 1: WARMUP TRAINING ({args.warmup_epochs} epochs) ===")
        logger.info(f"Starting warmup from epoch {start_epoch}")
        for epoch in range(start_epoch, args.warmup_epochs + 1):
            model.train()
            total_loss = 0
            start_time = time.time()
            logger.info(f"Epoch {epoch}/{args.warmup_epochs}: WARMUP - Clean training only")
            for i, (batch_funcs, batch_labels) in enumerate(tqdm(train_loader, desc=f"Warmup Epoch {epoch}")):
                batch_funcs = batch_funcs.to(device)
                batch_labels = batch_labels.to(device)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = model(batch_funcs, device=device)
                        loss = criterion(logits, batch_labels)
                else:
                    logits = model(batch_funcs, device=device)
                    loss = criterion(logits, batch_labels)
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * batch_labels.size(0)
            
            avg_loss = total_loss / len(train_loader.dataset)
            logger.info(f"Warmup Epoch {epoch}: Loss: {avg_loss:.4f}")
            
            # Validation during warmup
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_funcs, batch_labels in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                    batch_funcs = batch_funcs.to(device)
                    batch_labels = batch_labels.to(device)
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            logits = model(batch_funcs, device=device)
                            loss = criterion(logits, batch_labels)
                    else:
                        logits = model(batch_funcs, device=device)
                        loss = criterion(logits, batch_labels)
                    val_loss += loss.item() * batch_labels.size(0)
                    preds = (torch.sigmoid(logits) > 0.5).long()
                    correct += (preds == batch_labels.long()).sum().item()
                    total += batch_labels.size(0)
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total
            logger.info(f"Warmup Epoch {epoch}: Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")
            scheduler.step(avg_val_loss)
            
            # Save best models for each phase and overall
            is_best_warmup = False
            is_best = False
            
            if avg_val_loss < best_val_loss_warmup:
                best_val_loss_warmup = avg_val_loss
                best_epoch_warmup = epoch
                is_best_warmup = True
                torch.save(model.state_dict(), best_model_path_warmup)
                logger.info(f"Warmup Epoch {epoch}: Saved best model to {best_model_path_warmup}")
            
            # Save overall best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch_overall = epoch
                is_best = True
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Warmup Epoch {epoch}: Saved best overall model to {best_model_path}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_loss_warmup': best_val_loss_warmup,
                'best_val_loss_adv': best_val_loss_adv,
                'best_epoch_overall': best_epoch_overall,
                'best_epoch_warmup': best_epoch_warmup,
                'best_epoch_adv': best_epoch_adv,
            }, checkpoint_path)
            
            # Log to JSON (matching curriculum script schema)
            is_warmup_phase = True
            is_best_adv = False  # No adversarial phase in warmup
            log_entry = {
                'epoch': epoch,
                'train_total_loss': avg_loss,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'is_best': is_best,
                'is_best_warmup': is_best_warmup,
                'is_best_adv': is_best_adv,
                'is_warmup_phase': is_warmup_phase
            }
            log_data.append(log_entry)
            # Robustly write trainlog.json after every epoch
            with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(log_path)) as tf:
                json.dump(log_data, tf, indent=2)
                tempname = tf.name
            shutil.move(tempname, log_path)
    else:
        logger.info(f"Warmup already completed (resuming from epoch {start_epoch} > {args.warmup_epochs})")
    
    # Initialize adversarial examples (will be regenerated during adversarial training)
    adversarial_examples = {}
    
    # Phase 2: Adversarial training
    adv_start_epoch = max(start_epoch, args.warmup_epochs + 1)
    
    # AUTOMATICALLY LOAD BEST WARMUP MODEL BEFORE ADVERSARIAL TRAINING
    if not args.restart and adv_start_epoch <= args.max_epochs:
        best_warmup_path = os.path.join(args.results_dir, f'best_model_WARMUP_{ablation_name}.pt')
        if os.path.exists(best_warmup_path):
            logger.info(f"===== LOADING BEST WARMUP MODEL FOR ADVERSARIAL TRAINING =====")
            logger.info(f"Loading best warmup checkpoint: {best_warmup_path}")
            logger.info(f"  Best warmup val_loss: {best_val_loss_warmup:.4f} (epoch {best_epoch_warmup})")
            
            try:
                # Load the best warmup model state
                ckpt = torch.load(best_warmup_path, map_location=device)
                model.load_state_dict(ckpt)
                logger.info(f" Successfully loaded best warmup model weights")
                logger.info(f"  Adversarial training will start from optimal warmup state")
                logger.info(f"  This ensures adversarial training begins from the best clean performance")
            except Exception as e:
                logger.warning(f" Failed to load best warmup model: {e}")
                logger.warning(f"  Continuing with current model state for adversarial training")
        else:
            logger.info(f"===== NO BEST WARMUP MODEL FOUND =====")
            logger.info(f"Best warmup checkpoint not found: {best_warmup_path}")
            logger.info(f"  Adversarial training will start from current model state")
    elif args.restart:
        logger.info(f"===== RESTART MODE - SKIPPING BEST WARMUP LOAD =====")
        logger.info(f"--restart flag set: using current model state for adversarial training")
    else:
        logger.info(f"===== ADVERSARIAL TRAINING ALREADY COMPLETED =====")
        logger.info(f"Resuming from epoch {start_epoch} > {args.warmup_epochs}")
    if adv_start_epoch <= args.max_epochs:
        logger.info(f"=== PHASE 2: ADVERSARIAL TRAINING ({args.max_epochs - args.warmup_epochs} epochs) ===")
        logger.info(f"Starting adversarial training from epoch {adv_start_epoch}")
        for epoch in range(adv_start_epoch, args.max_epochs + 1):
            # Calculate curriculum lambda for adversarial training
            adv_epoch = epoch - args.warmup_epochs
            total_adv_epochs = args.max_epochs - args.warmup_epochs
            progress = adv_epoch / total_adv_epochs
            current_adv_lambda = args.adv_lambda_start + progress * (args.adv_lambda_end - args.adv_lambda_start)
            training_phase = "ADVERSARIAL"
            
            # Regenerate adversarial examples at first adversarial epoch and every 5 epochs
            regenerate_advs = False
            if adv_epoch == 1:  # First adversarial epoch
                regenerate_advs = True
                logger.info(f"===== GENERATING INITIAL ADVERSARIAL EXAMPLES AFTER WARMUP =====")
            elif adv_epoch % 5 == 1 and adv_epoch > 1:  # Every 5 epochs (6, 11, 16, etc.)
                regenerate_advs = True
                logger.info(f"===== REGENERATING ADVERSARIAL EXAMPLES (EPOCH {epoch}) =====")
                logger.info("Model has become more robust - generating fresh adversarial examples")
            
            if regenerate_advs:
                adversarial_examples = generate_and_save_adversarial_examples_hotflip(model, train_loader, args, device)
                if len(adversarial_examples) == 0:
                    logger.warning("No successful adversarial examples generated, continuing with clean training only")
                else:
                    logger.info("===== ADVERSARIAL EXAMPLES READY =====")
            
            model.train()
            total_clean_loss = 0
            total_adv_loss = 0
            total_loss = 0
            adv_batches = 0
            start_time = time.time()
            logger.info(f"Epoch {epoch}/{args.max_epochs}: {training_phase} - Clean + HotFlip Adversarial training (λ={current_adv_lambda:.3f})")
            for i, (batch_funcs, batch_labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
                # Move input to device before model call to avoid device mismatch
                batch_funcs = batch_funcs.to(device)
                batch_labels = batch_labels.to(device)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        clean_logits = model(batch_funcs, device=device)
                        clean_loss = criterion(clean_logits, batch_labels)
                else:
                    clean_logits = model(batch_funcs, device=device)
                    clean_loss = criterion(clean_logits, batch_labels)
                # Clean loss (always computed for full batch)
                total_clean_loss += clean_loss.item() * batch_labels.size(0)
                
                # Adversarial loss (only for successful adversarial examples)
                adv_loss = 0.0
                if i in adversarial_examples:
                    adv_batches += 1
                    adv_data = adversarial_examples[i]
                    adv_tokens = adv_data['adv_tokens'].to(device).long()
                    mal_labels = torch.ones(adv_tokens.size(0), dtype=torch.float, device=device)
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            adv_logits = model(adv_tokens, device=device)
                            adv_loss = criterion(adv_logits, mal_labels)
                    else:
                        adv_logits = model(adv_tokens, device=device)
                        adv_loss = criterion(adv_logits, mal_labels)
                    # FIXED: Scale adversarial loss by batch size to match clean loss scaling
                    total_adv_loss += adv_loss.item() * batch_labels.size(0)
                
                # Combined loss for optimization
                combined_loss = clean_loss + current_adv_lambda * adv_loss
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(combined_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    combined_loss.backward()
                    optimizer.step()
                
                # FIXED: Use combined_loss.item() * batch_labels.size(0) for consistent scaling
                total_loss += combined_loss.item() * batch_labels.size(0)
            avg_clean_loss = total_clean_loss / len(train_loader.dataset)
            avg_adv_loss = total_adv_loss / len(train_loader.dataset) if adv_batches > 0 else 0.0
            avg_total_loss = total_loss / len(train_loader.dataset)
            logger.info(f"Epoch {epoch}: Clean Loss: {avg_clean_loss:.4f}, Adv Loss: {avg_adv_loss:.4f}, Total Loss: {avg_total_loss:.4f}")
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_funcs, batch_labels in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                    # Move input to device before model call to avoid device mismatch
                    batch_funcs = batch_funcs.to(device)
                    batch_labels = batch_labels.to(device)
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            logits = model(batch_funcs, device=device)
                            loss = criterion(logits, batch_labels)
                    else:
                        logits = model(batch_funcs, device=device)
                        loss = criterion(logits, batch_labels)
                    val_loss += loss.item() * batch_labels.size(0)
                    preds = (torch.sigmoid(logits) > 0.5).long()
                    correct += (preds == batch_labels.long()).sum().item()
                    total += batch_labels.size(0)
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total
            logger.info(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")
            scheduler.step(avg_val_loss)

            # Save best models for each phase and overall
            is_best_adv = False
            is_best = False
            
            if avg_val_loss < best_val_loss_adv:
                best_val_loss_adv = avg_val_loss
                best_epoch_adv = epoch
                is_best_adv = True
                torch.save(model.state_dict(), best_model_path_adv)
                logger.info(f"Adversarial Epoch {epoch}: Saved best model to {best_model_path_adv}")
            
            # Save overall best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch_overall = epoch
                is_best = True
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Epoch {epoch}: Saved best overall model to {best_model_path}")

            # Save checkpoint with all bests
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_loss_warmup': best_val_loss_warmup,
                'best_val_loss_adv': best_val_loss_adv,
                'best_epoch_overall': best_epoch_overall,
                'best_epoch_warmup': best_epoch_warmup,
                'best_epoch_adv': best_epoch_adv,
            }, checkpoint_path)
            
            # Log to JSON (matching curriculum script schema)
            is_warmup_phase = False
            is_best_warmup = False  # Not in warmup phase
            log_entry = {
                'epoch': epoch,
                'train_total_loss': avg_total_loss,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'is_best': is_best,
                'is_best_warmup': is_best_warmup,
                'is_best_adv': is_best_adv,
                'is_warmup_phase': is_warmup_phase
            }
            if not is_warmup_phase:
                log_entry.update({
                    'train_clean_loss': avg_clean_loss,
                    'train_adv_loss': avg_adv_loss,
                    'adv_batches': adv_batches
                })
            log_data.append(log_entry)
            # Robustly write trainlog.json after every epoch
            with tempfile.NamedTemporaryFile('w', delete=False, dir=os.path.dirname(log_path)) as tf:
                json.dump(log_data, tf, indent=2)
                tempname = tf.name
            shutil.move(tempname, log_path)
    else:
        logger.info(f"Adversarial training already completed (resuming from epoch {start_epoch} > {args.max_epochs})")
    
    # Load the best overall model for final evaluation/deployment
    logger.info(f"Training complete: {args.max_epochs} total epochs")
    logger.info(f"Best overall model: {best_model_path} (val_loss={best_val_loss:.4f})")
    logger.info(f"Best warmup model: {best_model_path_warmup} (val_loss={best_val_loss_warmup:.4f})")
    logger.info(f"Best adversarial model: {best_model_path_adv} (val_loss={best_val_loss_adv:.4f})")
    
    # Load the best overall model into memory for immediate use
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f" Loaded best overall model from {best_model_path}")
        logger.info(f"  This model achieved val_loss={best_val_loss:.4f} at epoch {best_epoch_overall}")
    else:
        logger.warning(f" Best overall model file not found: {best_model_path}")
    
    logger.info("===== TRAINING COMPLETE =====")
    logger.info("Available models for evaluation:")
    logger.info(f"  1. Best overall: {best_model_path}")
    logger.info(f"  2. Best warmup: {best_model_path_warmup}")
    logger.info(f"  3. Best adversarial: {best_model_path_adv}")
    logger.info(f"  4. Final checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Training for Malware Classifier (HotFlip)")
    parser.add_argument('--data_glob', type=str, default='./data/*/disassembled/*.json',
                        help='Glob pattern for dataset JSON files')
    parser.add_argument('--use_attention', action='store_true', help='Use attention in binary aggregator')
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal', choices=['sinusoidal', 'learned', 'none'],
                        help='Type of positional encoding to use (sinusoidal, learned, or none)')
    parser.add_argument('--use_split_tokens', dest='use_split_tokens', action='store_true', help='Use atomic token splitting approach')
    parser.add_argument('--no-use_split_tokens', dest='use_split_tokens', action='store_false', help='Use merged tokens')
    parser.set_defaults(use_split_tokens=True)
    parser.add_argument('--use_boundaries', action='store_true', default=True, help='Add instruction boundary tokens')
    parser.add_argument('--vocab_path', type=str, default=None, help='Path to vocabulary file')
    parser.add_argument('--results_dir', type=str, default='.', help='Directory to save results')
    parser.add_argument('--ablation_name', type=str, default=None, help='Ablation name for logging')
    parser.add_argument('--max_func_len', type=int, default=64, help='Maximum function length')
    parser.add_argument('--max_funcs', type=int, default=64, help='Maximum number of functions per binary')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--min_freq', type=int, default=10, help='Minimum frequency threshold for vocabulary')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--max_epochs', type=int, default=30, help='Maximum number of training epochs (20 warmup + 10 adversarial)')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Number of warmup epochs before adversarial training')
    parser.add_argument('--scheduler_patience', type=int, default=2, help='Scheduler patience for learning rate reduction')
    parser.add_argument('--adv_lambda_start', type=float, default=0.4, help='Starting weight for adversarial loss (λ) in curriculum training (epoch 21)')
    parser.add_argument('--adv_lambda_end', type=float, default=0.8, help='Final weight for adversarial loss (λ) in curriculum training (epoch 30)')
    parser.add_argument('--restart', action='store_true', help='Force restart training from scratch (ignore existing checkpoints)')
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    try:
        main(args)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise 