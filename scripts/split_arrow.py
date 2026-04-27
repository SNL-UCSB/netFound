#!/usr/bin/env python3
"""
Split Arrow IPC stream files into N equal parts (by rows).

Usage:
    python split_arrow.py /path/to/dir/*.arrow          # default 5 parts
    python split_arrow.py -n 3 file1.arrow file2.arrow   # 3 parts
    python split_arrow.py -o /output/dir caida.arrow      # custom output dir

Output files are named <stem>_part1.arrow ... <stem>_partN.arrow
and placed next to the original (or in --output-dir if given).
"""

import argparse
import math
import os
import sys

import pyarrow as pa
import pyarrow.ipc as ipc


def split_file(file_path: str, n_parts: int, output_dir: str | None = None):
    """Read an Arrow IPC stream file and write N equal-row splits."""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    dest = output_dir or os.path.dirname(os.path.abspath(file_path))
    os.makedirs(dest, exist_ok=True)

    # ── Pass 1: read entire file into a single table ─────────────────────
    print(f"Reading {file_path} ...")
    batches = []
    with pa.OSFile(file_path, "rb") as source:
        reader = ipc.open_stream(source)
        schema = reader.schema
        for batch in reader:
            batches.append(batch)

    table = pa.Table.from_batches(batches, schema=schema)
    total_rows = table.num_rows
    rows_per_part = math.ceil(total_rows / n_parts)
    print(f"  Total rows: {total_rows:,}  →  {n_parts} parts of ~{rows_per_part:,} rows each")

    # ── Pass 2: slice and write each part ────────────────────────────────
    for i in range(n_parts):
        start = i * rows_per_part
        end = min(start + rows_per_part, total_rows)
        if start >= total_rows:
            break
        part = table.slice(start, end - start)

        out_name = f"{stem}_part{i + 1}.arrow"
        out_path = os.path.join(dest, out_name)

        with pa.OSFile(out_path, "wb") as sink:
            writer = ipc.new_stream(sink, schema)
            # Write in reasonably-sized batches (~5000 rows)
            batch_size = 5000
            for offset in range(0, part.num_rows, batch_size):
                chunk = part.slice(offset, min(batch_size, part.num_rows - offset))
                writer.write_table(chunk)
            writer.close()

        file_size = os.path.getsize(out_path)
        print(f"  Written {out_name}  ({end - start:,} rows, {file_size / 1e6:.1f} MB)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Split Arrow IPC stream files into N parts by rows.")
    parser.add_argument("files", nargs="+", help="Arrow files to split")
    parser.add_argument("-n", "--num-parts", type=int, default=5, help="Number of parts (default: 5)")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    for fpath in args.files:
        if not os.path.isfile(fpath):
            print(f"ERROR: File not found: {fpath}", file=sys.stderr)
            continue
        split_file(fpath, args.num_parts, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
