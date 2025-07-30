"""
Vocabulary Builder for Malware Classifier
----------------------------------------
Builds a vocabulary from disassembled malware dataset JSONs for use in training and testing.

This module provides functionality to:
- Tokenize instruction sequences into atomic components
- Build vocabulary from large datasets with progress tracking
- Normalize import tokens based on API importance
- Support both split and non-split tokenization approaches

Dependencies:
- Python 3.6+
- tqdm

Usage:
    python build_vocabulary.py --data_glob './data/*/disassembled/*.json' [other options]

See argument help below for all options.
"""
import os
import json
import re
from collections import Counter
from tqdm import tqdm

# Configuration constants
DEFAULT_MIN_FREQ = 10
DEFAULT_PROGRESS_SAVE_INTERVAL = 1000
DEFAULT_VOCAB_PATH = 'vocab.json'
DEFAULT_PROGRESS_PATH = 'vocab_progress.json'

# List of important API keywords (case-insensitive, partial match allowed)
# These APIs are commonly used in malware and should be preserved as specific tokens
IMPORTANT_APIS = [
    'Accept', 'AdjustTokenPrivileges', 'AttachThreadInput', 'Bind', 'BitBlt', 'CertOpenSystemStore',
    'Connect', 'ConnectNamedPipe', 'ControlService', 'CreateFile', 'CreateFileMapping', 'CreateMutex',
    'CreateProcess', 'CreateRemoteThread', 'CreateService', 'CreateToolhelp32Snapshot', 'CryptAcquireContext',
    'DeviceIoControl', 'EnableExecuteProtectionSupport', 'EnumProcesses', 'EnumProcessModules',
    'FindFirstFile', 'FindNextFile', 'FindResource', 'FindWindow', 'FtpPutFile', 'GetAdaptersInfo',
    'GetAsyncKeyState', 'GetDC', 'GetForegroundWindow', 'Gethostbyname', 'Gethostname', 'GetKeyState',
    'GetModuleFilename', 'GetModuleHandle', 'GetProcAddress', 'GetStartupInfo', 'GetSystemDefaultLangId',
    'GetTempPath', 'GetThreadContext', 'GetVersionEx', 'GetWindowsDirectory', 'inet_addr', 'InternetOpen',
    'InternetOpenUrl', 'InternetReadFile', 'InternetWriteFile', 'IsNTAdmin', 'IsWoW64Process', 'LdrLoadDll',
    'LoadResource', 'LsaEnumerateLogonSessions', 'MapViewOfFile', 'MapVirtualKey', 'Module32First',
    'Module32Next', 'NetScheduleJobAdd', 'NetShareEnum', 'NtQueryDirectoryFile', 'NtQueryInformationProcess',
    'NtSetInformationProcess', 'OpenMutex', 'OpenProcess', 'OutputDebugString', 'PeekNamedPipe',
    'Process32First', 'Process32Next', 'QueueUserAPC', 'ReadProcessMemory', 'Recv', 'RegisterHotKey',
    'RegOpenKey', 'ResumeThread', 'RtlCreateRegistryKey', 'RtlWriteRegistryValue', 'SamIConnect',
    'SamIGetPrivateData', 'SamQueryInformationUse', 'Send', 'SetFileTime', 'SetThreadContext',
    'SetWindowsHookEx', 'SfcTerminateWatcherThread', 'ShellExecute', 'StartServiceCtrlDispatcher',
    'SuspendThread', 'System', 'Thread32First', 'Thread32Next', 'Toolhelp32ReadProcessMemory',
    'URLDownloadToFile', 'VirtualAllocEx', 'VirtualProtectEx', 'WideCharToMultiByte', 'WinExec',
    'WriteProcessMemory', 'WSAStartup'
]

# Compile regex for suspicious characters
SUSPICIOUS_CHARS = re.compile(r'[@?$]')

def normalize_import_token(token):
    """
    Helper to check if a token is an import and should be normalized.
    Special character check takes precedence.
    """
    # Only process tokens that look like imports
    if 'IMPORT' in token:
        parts = token.split('|')
        api = parts[1] if len(parts) > 1 else ''
        # If suspicious chars, treat as generic import (takes precedence)
        if SUSPICIOUS_CHARS.search(api):
            return 'IMPORT|GENERIC'
        # If matches important API, keep as IMPORT|API
        for important in IMPORTANT_APIS:
            if important.lower() in api.lower():
                return f'IMPORT|{important}'
        # Otherwise, treat as generic import
        return 'IMPORT|GENERIC'
    return token

def split_instruction_token(token):
    """
    Split instruction tokens into atomic components (opcode + operands).
    
    Examples:
    - "mov|REG,REG" -> ["mov", "REG", "REG"]
    - "call|IMM" -> ["call", "IMM"]
    - "IMPORT|CreateFile" -> ["IMPORT", "CreateFile"]
    
    Benefits:
    - Smaller vocabulary size
    - Better generalization across operand patterns
    - Improved attention interpretability
    """
    # Handle import tokens specially
    if token.startswith('IMPORT|'):
        parts = token.split('|', 1)
        return [parts[0], parts[1]] if len(parts) == 2 else [token]
    
    # Handle regular instruction tokens
    if '|' in token:
        parts = token.split('|')
        opcode = parts[0]
        operands = parts[1] if len(parts) > 1 else ''
        
        tokens = [opcode]
        
        # Split operands by comma and add each as separate token
        if operands:
            operand_list = [op.strip() for op in operands.split(',') if op.strip()]
            tokens.extend(operand_list)
        
        return tokens
    
    # Return as-is if no splitting needed
    return [token]

def tokenize_instruction_sequence(instructions, use_boundaries=True):
    """
    Convert a sequence of instructions into atomic tokens.
    
    Args:
        instructions: List of instruction strings
        use_boundaries: Whether to add instruction boundary tokens
    
    Returns:
        List of atomic tokens
    """
    tokens = []
    
    for instr in instructions:
        # Normalize import tokens first
        normalized = normalize_import_token(instr)
        
        # Split into atomic components
        atomic_tokens = split_instruction_token(normalized)
        tokens.extend(atomic_tokens)
        
        # Add instruction boundary token if requested
        if use_boundaries:
            tokens.append('<EOL>')
    
    # Remove trailing boundary token
    if use_boundaries and tokens and tokens[-1] == '<EOL>':
        tokens.pop()
    
    return tokens

def build_vocab(file_list, min_freq=10, use_split_tokens=True, use_boundaries=True, progress_path='vocab_progress.json'):
    """
    Build vocabulary from all files with improved token splitting.
    
    Args:
        file_list: List of JSON file paths to process
        min_freq: Minimum frequency threshold for including tokens
        use_split_tokens: Whether to split instructions into atomic tokens
        use_boundaries: Whether to add instruction boundary tokens
        progress_path: Path to save/load progress
    
    Returns:
        List of vocabulary tokens ordered by frequency
    """
    token_counter = Counter()
    processed_files = set()
    
    # Load progress if exists
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as pf:
            try:
                processed_files = set(json.load(pf))
            except Exception as e:
                print(f"[ERROR] Could not load progress file: {e}")
    
    total_files = len(file_list)
    
    for idx, file in enumerate(tqdm(file_list, desc="Building vocab")):
        if file in processed_files:
            continue
            
        with open(file, 'r') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Could not read {file}: {e}")
                continue
                
            if not isinstance(data, dict):
                print(f"[ERROR] File {file} is not a dict, got {type(data)}. Skipping.")
                continue
            
            for func in data.get('functions', []):
                instructions = func.get('instructions', [])
                
                if use_split_tokens:
                    # Use new atomic tokenization approach
                    tokens = tokenize_instruction_sequence(instructions, use_boundaries)
                    for token in tokens:
                        token_counter[token] += 1
                else:
                    # Use original approach
                    for instr in instructions:
                        norm = normalize_import_token(instr)
                        token_counter[norm] += 1
        
        processed_files.add(file)
        
        # Save progress every 1000 files
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_files:
            with open(progress_path, 'w') as pf:
                json.dump(list(processed_files), pf)
    
    # Remove progress file after completion
    if os.path.exists(progress_path):
        os.remove(progress_path)
    
    # Only keep tokens with frequency >= min_freq, and order by frequency (most common first)
    vocab = [tok for tok, freq in token_counter.most_common() if freq >= min_freq]
    
    # Add special tokens
    special_tokens = ['<UNK>', '<PAD>']
    if use_boundaries:
        special_tokens.append('<EOL>')
    
    # Add special tokens at the end to maintain frequency ordering for main vocabulary
    vocab.extend(special_tokens)
    
    return vocab

def save_vocab(vocab, vocab_path='vocab.json'):
    """Save vocabulary to JSON file."""
    with open(vocab_path, 'w') as vf:
        json.dump(vocab, vf)
    print(f"[INFO] Saved vocabulary to {vocab_path}")

def load_vocab(vocab_path='vocab.json'):
    """Load vocabulary from JSON file."""
    with open(vocab_path, 'r') as vf:
        vocab = json.load(vf)
    print(f"[INFO] Loaded vocabulary from {vocab_path}")
    return vocab

def print_vocab_stats(vocab, token_counter=None):
    """Print vocabulary statistics."""
    print(f"[INFO] Vocabulary size: {len(vocab)}")
    
    if token_counter:
        total_tokens = sum(token_counter.values())
        print(f"[INFO] Total tokens processed: {total_tokens:,}")
        
        # Show most common tokens
        print(f"[INFO] Top 20 most frequent tokens:")
        for i, (token, freq) in enumerate(token_counter.most_common(20)):
            print(f"  {i+1:2d}. {token:20s} : {freq:,}")

if __name__ == "__main__":
    import glob
    import argparse
    
    parser = argparse.ArgumentParser(description='Build vocabulary from malware dataset')
    parser.add_argument('--data_glob', type=str, default='./data/*/disassembled/*.json',
                        help='Glob pattern for dataset JSON files')
    parser.add_argument('--min_freq', type=int, default=10,
                        help='Minimum frequency threshold for including tokens')
    parser.add_argument('--use_split_tokens', action='store_true', default=True,
                        help='Use atomic token splitting approach')
    parser.add_argument('--use_boundaries', action='store_true', default=True,
                        help='Add instruction boundary tokens')
    parser.add_argument('--vocab_path', type=str, default='vocab.json',
                        help='Output path for vocabulary file')
    
    args = parser.parse_args()
    
    # Get all files
    all_files = sorted(glob.glob(args.data_glob))
    print(f"Found {len(all_files)} files")
    
    # Build vocabulary
    vocab = build_vocab(
        all_files, 
        min_freq=args.min_freq,
        use_split_tokens=args.use_split_tokens,
        use_boundaries=args.use_boundaries
    )
    
    # Save vocabulary
    save_vocab(vocab, args.vocab_path)
    print_vocab_stats(vocab) 