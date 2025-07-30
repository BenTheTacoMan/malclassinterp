"""
Function and IAT Byte Extractor for PE Files
-------------------------------------------
Extracts function bytes and import address table (IAT) from PE binaries using DeepDiCore.

Dependencies:
- Python 3.6+
- pefile
- numpy
- DeepDiCore (shared object must be available and path provided)

Usage:
    python byteExtractor.py --input_dir binaries/ --output_dir output/ --deepdi_path path/to/DeepDiCore.so
"""
import os
import json
import pefile
import numpy as np
import importlib.util
from pathlib import Path
import argparse

# Configuration
# Must insert user key provided by DeepDi: key = ""
gpu = True
batch_size = 1024 * 1024
label = 1

def load_deepdi(deepdi_path):
    """Dynamically load DeepDiCore shared object."""
    if not Path(deepdi_path).exists():
        raise FileNotFoundError(f"DeepDiCore shared object not found at {deepdi_path}")
    spec = importlib.util.spec_from_file_location("DeepDiCore", deepdi_path)
    DeepDiCore = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DeepDiCore)
    return DeepDiCore

def va_to_file_offset(pe, va):
    """Convert a virtual address to a file offset in the PE file."""
    rva = va - pe.OPTIONAL_HEADER.ImageBase
    try:
        return pe.get_offset_from_rva(rva)
    except Exception:
        return None

def extract_iat_map(pe):
    """Extract the Import Address Table (IAT) mapping and a list of imports."""
    iat_map = {}
    iat_list = []
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll_name = entry.dll.decode(errors='ignore')
            for imp in entry.imports:
                if imp.address and imp.name:
                    func_name = imp.name.decode(errors='ignore')
                    iat_map[imp.address] = func_name
                    iat_list.append(f"{dll_name}.{func_name}")
                elif imp.address:
                    iat_list.append(f"{dll_name}.#{imp.ordinal}")
    return iat_map, iat_list

def extract_text_section(pe):
    """Extract executable code sections from the PE file."""
    code_bytes = b""
    addr_map = []
    for section in pe.sections:
        if section.IMAGE_SCN_MEM_EXECUTE:
            data = section.get_data()
            start_va = pe.OPTIONAL_HEADER.ImageBase + section.VirtualAddress
            addr_map.append((start_va, len(data)))
            code_bytes += data
    return code_bytes, addr_map

def create_addr_mapping(code_addr):
    """Create a mapping from code offsets to virtual addresses."""
    total_len = sum(length for _, length in code_addr)
    addr_mapping = np.empty(total_len, dtype=np.int64)
    cur = 0
    for start, length in code_addr:
        addr_mapping[cur:cur + length] = np.arange(start=start, stop=start + length, step=1, dtype=np.int64)
        cur += length
    return addr_mapping

def process_binary(file_path, disasm):
    """Process a single PE binary, extracting function bytes and IAT."""
    pe = pefile.PE(str(file_path), fast_load=True)
    pe.parse_data_directories()  # Ensure import table is parsed
    x64 = not pe.FILE_HEADER.IMAGE_FILE_32BIT_MACHINE
    code, code_addr = extract_text_section(pe)
    if len(code) == 0:
        return {"file_name": file_path.name, "label": label, "functions": [], "imports": [], "iat_map": {}}
    addr_mapping = create_addr_mapping(code_addr)
    iat_map, iat_list = extract_iat_map(pe)
    functions = []
    for i in range(0, len(code), batch_size):
        end_idx = min(i + batch_size, len(code))
        chunk = code[i:end_idx]
        disasm.Disassemble(chunk, x64)
        disasm.Sync()
        inst_pred = disasm.GetInstructionProb() >= 0.5
        inst_len = disasm.GetInstructionLength()[inst_pred]
        func_pred = disasm.GetFunction()
        chunk_addr_mapping = addr_mapping[i:end_idx]
        inst_addr = chunk_addr_mapping[inst_pred]
        func_addr = sorted(set(chunk_addr_mapping[func_pred]))
        func_set = set(func_addr)
        sorted_indices = np.argsort(inst_addr)
        current_func = None
        func_bytes = b""
        for idx in sorted_indices:
            addr = inst_addr[idx]
            if addr in func_set:
                if current_func is not None and func_bytes:
                    functions.append({
                        "func_start": int(current_func),
                        "bytes": func_bytes.hex()
                    })
                current_func = addr
                func_bytes = b""
            if current_func is not None:
                length = int(inst_len[idx])
                file_offset = va_to_file_offset(pe, addr)
                if file_offset is not None:
                    instr_bytes = pe.__data__[file_offset:file_offset + length]
                    func_bytes += instr_bytes
        if current_func is not None and func_bytes:
            functions.append({
                "func_start": int(current_func),
                "bytes": func_bytes.hex()
            })
    return {
        "file_name": file_path.name,
        "label": label,
        "functions": functions,
        "imports": iat_list,
        "iat_map": {str(k): v for k, v in iat_map.items()}
    }

def main():
    parser = argparse.ArgumentParser(description="Extract function bytes and IAT from PE files.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with binaries')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for JSON files')
    parser.add_argument('--deepdi_path', type=str, required=True, help='Path to DeepDiCore shared object (e.g., DeepDiCore.so)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_path = output_dir / "processed.json"
    if processed_path.exists():
        with open(processed_path, "r") as f:
            processed = set(json.load(f))
    else:
        processed = set()

    files = [file for file in input_dir.iterdir() if file.suffix.lower() in [".exe", ".dll"]]

    DeepDiCore = load_deepdi(args.deepdi_path)
    disasm = DeepDiCore.Disassembler(key, gpu)

    for file in files:
        file_str = str(file.resolve())
        if file_str in processed:
            print(f"Skipping already processed: {file.name}")
            continue
        print(f"Processing {file.name}")
        try:
            result = process_binary(file, disasm)
            with open(output_dir / f"{file.name}.json", "w") as f:
                json.dump(result, f, indent=2)
            processed.add(file_str)
            with open(processed_path, "w") as f:
                json.dump(list(processed), f, indent=2)
        except Exception as e:
            print(f"Failed to process {file.name}: {e}")

if __name__ == "__main__":
    main() 