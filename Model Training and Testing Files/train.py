"""
Malware Classifier Training Script
---------------------------------
Trains a malware classifier using function-level features from disassembled binaries.

Dependencies:
- Python 3.6+
- torch
- numpy
- tqdm
- model.py, dataset.py (must be present in the same directory)

Usage:
    python train.py --data_glob './data/*/disassembled/*.json' [other options]
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

def set_all_seeds(seed):
    """Set seeds for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def collate_fn(batch, tokenizer, max_func_len=64, max_funcs=64):
    return collate_batch(batch, tokenizer, max_func_len, max_funcs)

def main(args):
    seed = getattr(args, 'seed', 42)
    set_all_seeds(seed)
    print(f"Set all random seeds to {seed}")
    
    ablation_name = args.ablation_name or f"pos-{args.pos_encoding}_attn-{'attn' if args.use_attention else 'mean'}"
    print(f"Positional Encoding: {args.pos_encoding}, Aggregator: {'Attention' if args.use_attention else 'Mean'}")
    print(f"Ablation Name: {ablation_name}")

    print("CUDA available:", torch.cuda.is_available())
    print("torch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))

    data_glob = args.data_glob
    all_files = sorted(glob.glob(data_glob))
    print(f"Found {len(all_files)} files before filtering.")
    
    no_func_path = os.path.join(os.path.dirname(__file__), 'no_functions_files.txt')
    if os.path.exists(no_func_path):
        with open(no_func_path, 'r') as nf:
            no_func_files = set(line.strip().replace('\\', '/') for line in nf if line.strip())
        def filter_no_func(files):
            return [f for f in files if f.replace('\\', '/') not in no_func_files]
        all_files = filter_no_func(all_files)
        print(f"After filtering: {len(all_files)} files remain.")
    else:
        print(f"Warning: {no_func_path} not found. No files filtered.")
    
    train_files, val_files, test_files = split_dataset(all_files, seed=seed)
    assert set(train_files).isdisjoint(test_files), "Train/Test overlap!"
    assert set(val_files).isdisjoint(test_files),   "Val/Test overlap!"
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    if args.vocab_path:
        vocab_path = args.vocab_path
        print(f"Using provided vocab file: {vocab_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file {vocab_path} does not exist!")
        opcode_vocab = load_vocab(vocab_path)
    else:
        ablation_name = f"pos-{args.pos_encoding}_attn-{'attn' if args.use_attention else 'mean'}"
        vocab_path = os.path.join(args.results_dir, f'vocab_{ablation_name}.json')
        print(f"Building vocab at: {vocab_path}")
        min_freq = getattr(args, 'min_freq', 10)
        print(f"Building vocabulary with min_freq={min_freq}")
        opcode_vocab = build_vocab(
            train_files, 
            min_freq=min_freq, 
            use_split_tokens=args.use_split_tokens,
            use_boundaries=args.use_boundaries
        )
        save_vocab(opcode_vocab, vocab_path)
    print(f"Vocabulary size: {len(opcode_vocab)}")

    from model import OpcodeTokenizer
    tokenizer = OpcodeTokenizer(opcode_vocab)
    max_func_len = args.max_func_len
    max_funcs = args.max_funcs
    batch_size = args.batch_size
    print(f"max_func_len={max_func_len}, max_funcs={max_funcs}, batch_size={batch_size}")
    
    train_dataset = MalwareDataset(train_files, tokenizer, max_func_len, max_funcs, cache_in_memory=False, 
                                  use_split_tokens=args.use_split_tokens, use_boundaries=args.use_boundaries, seed=seed)
    val_dataset = MalwareDataset(val_files, tokenizer, max_func_len, max_funcs, cache_in_memory=False,
                                use_split_tokens=args.use_split_tokens, use_boundaries=args.use_boundaries, seed=seed)
    test_dataset = MalwareDataset(test_files, tokenizer, max_func_len, max_funcs, cache_in_memory=False,
                                 use_split_tokens=args.use_split_tokens, use_boundaries=args.use_boundaries, seed=seed)

    collate = partial(collate_fn, tokenizer=tokenizer, max_func_len=max_func_len, max_funcs=max_funcs)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, 
                             num_workers=16, pin_memory=True, persistent_workers=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, 
                           num_workers=16, pin_memory=True, persistent_workers=True)

    sample_batch = next(iter(train_loader))
    sample_funcs, sample_labels = sample_batch

    device = get_device()
    from model import MalwareClassifier
    model = MalwareClassifier(opcode_vocab, d_model=256, nhead=8, num_layers=2, max_func_len=max_func_len, max_funcs=max_funcs, dropout=0.2, use_attention=args.use_attention, pos_encoding=args.pos_encoding).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size()}")
    print(f"Embedding num_embeddings: {model.func_encoder.embedding.num_embeddings}")
    if tokenizer.vocab_size() != model.func_encoder.embedding.num_embeddings:
        print("ERROR: Tokenizer and embedding dimension mismatch!")
        raise RuntimeError("Tokenizer and embedding dimension mismatch!")
    max_token_id = sample_funcs.max().item()
    emb_slots = model.func_encoder.embedding.num_embeddings
    if max_token_id >= emb_slots:
        raise RuntimeError(f"Token ID {max_token_id} >= embedding rows {emb_slots}. This will cause CUDA assertion errors!")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    pos_weight = torch.tensor([1.0], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    checkpoint_path = os.path.join(args.results_dir, f'checkpoint_{ablation_name}.pt')
    best_model_path = os.path.join(args.results_dir, f'best_model_{ablation_name}.pt')
    log_path = args.log_path or os.path.join(args.results_dir, f'trainlog_{ablation_name}.json')
    start_epoch = 1
    best_val_loss = float('inf')
    log_data = []
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    for epoch in range(start_epoch, 20):
        model.train()
        total_loss = 0
        start_time = time.time()
        print(f"\n[Epoch {epoch}] Training...")
        data_times = []
        compute_times = []
        batch_start = time.time()
        for i, (batch_funcs, batch_labels) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
            data_end = time.time()
            data_times.append(data_end - batch_start)
            compute_start = time.time()
            batch_labels = batch_labels.to(device)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(batch_funcs, device=device)
                    loss = criterion(logits, batch_labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_funcs, device=device)
                loss = criterion(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            compute_end = time.time()
            compute_times.append(compute_end - compute_start)
            total_loss += loss.item() * batch_labels.size(0)
            batch_start = time.time()
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        print(f"[Epoch {epoch}] Validating...")
        with torch.no_grad():
            for batch_funcs, batch_labels in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
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
        print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")
        print(f"[Epoch {epoch}] Time elapsed: {time.time() - start_time:.2f} seconds")
        scheduler.step(avg_val_loss)
        is_best = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[Epoch {epoch}] Saved best model to {best_model_path}.")
            is_best = True
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_path)
        print(f"[Epoch {epoch}] Checkpoint saved to {checkpoint_path}")
        log_data.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'is_best': is_best
        })
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    print("Training complete. Running test_model.py...")
    test_model_path = shutil.which('python') or 'python'
    test_args = [test_model_path, os.path.join(os.path.dirname(__file__), 'test_model.py'),
                 '--data_glob', args.data_glob,
                 '--use_attention' if args.use_attention else '',
                 '--pos_encoding', args.pos_encoding,
                 '--vocab_path', vocab_path,
                 '--ablation_name', args.ablation_name,
                 '--log_path', args.log_path,
                 '--results_dir', args.results_dir]
    test_args = [a for a in test_args if a]
    result = subprocess.run(test_args, capture_output=True, text=True)
    print(result.stdout)
    test_acc = None
    for line in result.stdout.splitlines():
        if 'Test Accuracy:' in line:
            try:
                test_acc = float(line.split('Test Accuracy:')[1].strip().split()[0])
            except Exception:
                pass
    if test_acc is not None:
        log_data.append({'test_acc': test_acc, 'ablation': args.ablation_name, 'best_model': best_model_path})
        with open(args.log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
    print(f"Ablation results saved to {args.log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a malware classifier on disassembled binary features.")
    parser.add_argument('--data_glob', type=str, default='./data/*/disassembled/*.json',
                        help='Glob pattern for dataset JSON files')
    parser.add_argument('--use_attention', action='store_true', help='Use attention in binary aggregator')
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal', choices=['sinusoidal', 'learned', 'none'],
                        help='Type of positional encoding to use (sinusoidal, learned, or none)')
    parser.add_argument('--use_split_tokens', dest='use_split_tokens', action='store_true', help='Use atomic token splitting approach (recommended for better generalization)')
    parser.add_argument('--no-use_split_tokens', dest='use_split_tokens', action='store_false', help='Use merged tokens (original approach)')
    parser.set_defaults(use_split_tokens=True)
    parser.add_argument('--use_boundaries', action='store_true', default=True,
                        help='Add instruction boundary tokens (<EOL>)')
    parser.add_argument('--vocab_path', type=str, default=None, help='Path to vocabulary file (overrides ablation default)')
    parser.add_argument('--results_dir', type=str, default='.', help='Directory to save results for this ablation')
    parser.add_argument('--ablation_name', type=str, default=None, help='Ablation name for logging and file naming')
    parser.add_argument('--log_path', type=str, default=None, help='Path to training log JSON file')
    parser.add_argument('--max_func_len', type=int, default=64, help='Maximum function length (sequence length)')
    parser.add_argument('--max_funcs', type=int, default=64, help='Maximum number of functions per binary')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--min_freq', type=int, default=10, help='Minimum frequency threshold for vocabulary (lower=bigger vocab, higher=smaller vocab)')
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    main(args) 