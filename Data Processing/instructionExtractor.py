"""
Instruction Abstraction Extractor
--------------------------------
Disassembles function bytes from JSONs (produced by byteExtractor.py) and abstracts instructions using Capstone.

Dependencies:
- Python 3.6+
- capstone

Usage:
    python instructionExtractor.py --input_dir input_jsons/ --output_dir output_jsons/ --workers 8
"""
import os
import json
import capstone
from pathlib import Path
import argparse
import multiprocessing

def extract_iat_map_from_imports(imports):
    """Convert imports list to a simple lookup for instruction abstraction."""
    return {imp.split('.')[-1]: imp for imp in imports}

def operand_to_token(insn, op, iat_map):
    """Abstract a single operand for instruction abstraction."""
    if insn.mnemonic.startswith("call") or insn.mnemonic.startswith("jmp"):
        if op.type == capstone.CS_OP_IMM:
            func_name = iat_map.get(str(op.imm))
            if func_name:
                return f"[IMPORT]_{func_name}"
        elif op.type == capstone.CS_OP_MEM and op.mem.disp != 0:
            func_name = iat_map.get(str(op.mem.disp))
            if func_name:
                return f"[IMPORT]_{func_name}"
    if op.type == capstone.CS_OP_REG:
        return "REG"
    elif op.type == capstone.CS_OP_IMM:
        return "IMM"
    elif op.type == capstone.CS_OP_MEM:
        if op.mem.base != 0 and op.mem.disp != 0:
            return "MEM_REG_OFFSET"
        elif op.mem.base != 0:
            return "MEM_REG"
        else:
            return "MEM_CONST"
    else:
        return "OP"

def instruction_to_token(insn, iat_map):
    """Convert a disassembled instruction to an abstracted token string."""
    ops = [operand_to_token(insn, op, iat_map) for op in insn.operands]
    return f"{insn.mnemonic}|{','.join(ops)}" if ops else insn.mnemonic

def process_file(json_path, output_dir, processed_set, processed_path, lock):
    file_str = str(json_path.resolve())
    if file_str in processed_set:
        print(f"Skipping already processed: {json_path.name}")
        return
    print(f"Processing file: {json_path.name}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    imports = data.get("imports", [])
    iat_map = data.get("iat_map", {})
    functions = []
    for func in data.get("functions", []):
        func_start = func["func_start"]
        raw_bytes = bytes.fromhex(func["bytes"])
        instructions = []
        for mode in [capstone.CS_MODE_64, capstone.CS_MODE_32]:
            try:
                md = capstone.Cs(capstone.CS_ARCH_X86, mode)
                md.detail = True
                for insn in md.disasm(raw_bytes, func_start):
                    instructions.append(instruction_to_token(insn, iat_map))
                if instructions:
                    break
            except Exception:
                continue
        functions.append({
            "func_start": func_start,
            "bytes": func["bytes"],
            "instructions": instructions
        })
    out_data = {
        "file_name": data["file_name"],
        "label": data["label"],
        "functions": functions,
        "imports": imports,
        "iat_map": iat_map
    }
    out_path = output_dir / f"{data['file_name']}.json"
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f"Finished processing file: {json_path.name}")
    # Update processed.json safely
    with lock:
        processed_set.add(file_str)
        with open(processed_path, 'w') as pf:
            json.dump(list(processed_set), pf, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Disassemble function bytes and abstract instructions using Capstone.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with byteExtractor JSONs')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write output JSONs')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
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

    json_files = [f for f in input_dir.iterdir() if f.suffix == '.json' and f.name != 'processed.json']
    lock = multiprocessing.Manager().Lock()
    from functools import partial
    process_func = partial(process_file, output_dir=output_dir, processed_set=processed, processed_path=processed_path, lock=lock)
    with multiprocessing.Pool(args.workers) as pool:
        pool.map(process_func, json_files)

if __name__ == "__main__":
    main() 