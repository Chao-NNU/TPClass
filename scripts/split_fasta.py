#!/usr/bin/env python3
"""
split_fasta.py <input.fasta> <chunk_size>
按 chunk_size 条 FASTA 拆成多份（_000.fasta, _001.fasta, ...）。
用于 TClass 在多卡上并行。
"""
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: split_fasta.py <input.fasta> <chunk_size> [output_prefix]")
        sys.exit(1)
    inp = Path(sys.argv[1])
    chunk = int(sys.argv[2])
    prefix = sys.argv[3] if len(sys.argv) > 3 else inp.with_suffix("").as_posix() + "_chunk"

    n = 0
    part = 0
    f = None
    with open(inp) as src:
        for line in src:
            if line.startswith(">"):
                if n > 0 and n % chunk == 0:
                    f.close()
                    part += 1
                if n % chunk == 0:
                    out = f"{prefix}_{part:03d}.fasta"
                    f = open(out, "w")
                    print(f"  writing {out} (entries {n+1}..)")
                n += 1
            f.write(line)
    if f:
        f.close()
    print(f"Done. {n} entries split into {part+1} chunks, prefix={prefix}")

if __name__ == "__main__":
    main()
