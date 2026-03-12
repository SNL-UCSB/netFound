#!/usr/bin/env python3
"""
Describe an Arrow dataset file used for network flow pre-training / fine-tuning.

Outputs to console:
  - Schema & basic file info
  - Packets per flow distribution
  - Flow duration distribution
  - Bytes per packet distribution
  - Inter-arrival time (IAT) distribution
  - Total bytes per flow distribution
  - Protocol breakdown
  - Direction (fwd/bwd) ratio
  - Label (source file) breakdown

Usage:
    python describe_arrow.py <arrow_file> [arrow_file2 ...]

Compare multiple datasets by passing several files.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


# ── helpers ──────────────────────────────────────────────────────────────────

PROTO_NAMES = {6: "TCP", 17: "UDP", 1: "ICMP", 2: "IGMP", 47: "GRE", 50: "ESP"}


def fmt_duration(us: float) -> str:
    """Format microseconds into a human-readable string."""
    if us < 1_000:
        return f"{us:.1f} µs"
    elif us < 1_000_000:
        return f"{us / 1_000:.2f} ms"
    else:
        return f"{us / 1_000_000:.3f} s"


def percentile_line(arr: np.ndarray, label: str = "", fmt_fn=None) -> str:
    """Return a one-line percentile summary."""
    if len(arr) == 0:
        return f"  {label}: (empty)"
    pcts = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    vals = np.percentile(arr, pcts)
    fn = fmt_fn or (lambda x: f"{x:,.1f}")
    parts = [f"p{p}={fn(v)}" for p, v in zip(pcts, vals)]
    return f"  {label}  mean={fn(arr.mean())}  std={fn(arr.std())}  | " + "  ".join(parts)


def section(title: str):
    width = 80
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def histogram_ascii(arr: np.ndarray, bins: int = 20, width: int = 50, label: str = ""):
    """Print a compact ASCII histogram."""
    if len(arr) == 0:
        print("  (no data)")
        return
    counts, edges = np.histogram(arr, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1
    print(f"  {label} (n={len(arr):,})")
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar_len = int(c / max_count * width)
        bar = "█" * bar_len
        print(f"    [{lo:>12.1f}, {hi:>12.1f}) | {bar} {c:,}")
    print()


# ── data loading ─────────────────────────────────────────────────────────────

def load_arrow(file_path: str):
    """
    Read all batches from an Arrow IPC stream file and return numpy arrays
    for each statistic of interest.
    """
    flow_durations = []
    packets_per_flow = []
    bytes_per_packet = []
    total_bytes_per_flow = []
    iats_all = []
    protocols = []
    fwd_counts = []
    bwd_counts = []
    labels = []
    counts_per_flow = []
    burst_lengths = []

    num_batches = 0
    total_rows = 0

    with pa.OSFile(file_path, "rb") as source:
        reader = ipc.open_stream(source)
        schema = reader.schema

        for batch in reader:
            num_batches += 1
            n = batch.num_rows
            total_rows += n

            # ── flow_duration ────
            if "flow_duration" in schema.names:
                flow_durations.extend(batch.column("flow_duration").to_pylist())

            # ── directions → packets per flow & fwd/bwd split ────
            if "directions" in schema.names:
                dirs = batch.column("directions").to_pylist()
                for d in dirs:
                    pkt_count = len(d)
                    packets_per_flow.append(pkt_count)
                    fwd = sum(1 for x in d if x)
                    fwd_counts.append(fwd)
                    bwd_counts.append(pkt_count - fwd)

            # ── bytes ────
            if "bytes" in schema.names:
                byte_lists = batch.column("bytes").to_pylist()
                for bl in byte_lists:
                    bytes_per_packet.extend(bl)
                    total_bytes_per_flow.append(sum(bl))

            # ── iats ────
            if "iats" in schema.names:
                iat_lists = batch.column("iats").to_pylist()
                for il in iat_lists:
                    iats_all.extend(il)

            # ── protocol ────
            if "protocol" in schema.names:
                protocols.extend(batch.column("protocol").to_pylist())

            # ── labels ────
            if "labels" in schema.names:
                labels.extend(batch.column("labels").to_pylist())

            # ── counts ────
            if "counts" in schema.names:
                cnt_lists = batch.column("counts").to_pylist()
                for cl in cnt_lists:
                    counts_per_flow.append(sum(cl))

            # ── burst_tokens ────
            if "burst_tokens" in schema.names:
                bt_lists = batch.column("burst_tokens").to_pylist()
                for bt in bt_lists:
                    burst_lengths.append(len(bt))

    return {
        "schema": schema,
        "num_batches": num_batches,
        "total_rows": total_rows,
        "flow_durations": np.array(flow_durations, dtype=np.float64),
        "packets_per_flow": np.array(packets_per_flow, dtype=np.int64),
        "bytes_per_packet": np.array(bytes_per_packet, dtype=np.int64),
        "total_bytes_per_flow": np.array(total_bytes_per_flow, dtype=np.int64),
        "iats_all": np.array(iats_all, dtype=np.float64),
        "protocols": protocols,
        "fwd_counts": np.array(fwd_counts, dtype=np.int64),
        "bwd_counts": np.array(bwd_counts, dtype=np.int64),
        "labels": labels,
        "counts_per_flow": np.array(counts_per_flow, dtype=np.int64),
        "burst_lengths": np.array(burst_lengths, dtype=np.int64),
    }


# ── reporting ────────────────────────────────────────────────────────────────

def report(file_path: str, d: dict):
    file_size = os.path.getsize(file_path)
    name = os.path.basename(file_path)

    section(f"Dataset: {name}")
    print(f"  Path        : {file_path}")
    print(f"  File size   : {file_size / 1e6:.2f} MB  ({file_size / 1e9:.3f} GB)")
    print(f"  Batches     : {d['num_batches']:,}")
    print(f"  Total flows : {d['total_rows']:,}")
    print()
    print("  Schema:")
    for i in range(len(d["schema"])):
        f = d["schema"].field(i)
        print(f"    {f.name:20s} : {f.type}")

    # ── Packets per flow ─────────────────────────────────────────────────
    section("Packets per Flow")
    ppf = d["packets_per_flow"]
    if len(ppf) > 0:
        print(percentile_line(ppf, "pkts/flow", fmt_fn=lambda x: f"{x:.1f}"))
        print()
        # Value counts for small packet counts
        vc = Counter(ppf.tolist())
        print("  Top packet-count values:")
        for val, cnt in sorted(vc.items(), key=lambda x: -x[1])[:15]:
            pct = cnt / len(ppf) * 100
            print(f"    {val:>6d} pkts : {cnt:>10,} flows  ({pct:5.2f}%)")
        print()
        # Clipped histogram for better visibility
        clip_hi = float(np.percentile(ppf, 99))
        histogram_ascii(ppf[ppf <= clip_hi], bins=20, label="Packets/flow (clipped at p99)")

    # ── Flow duration ────────────────────────────────────────────────────
    section("Flow Duration (µs)")
    dur = d["flow_durations"]
    if len(dur) > 0:
        print(percentile_line(dur, "duration", fmt_fn=fmt_duration))
        print()
        nonzero = dur[dur > 0]
        print(f"  Zero-duration flows: {np.sum(dur == 0):,} / {len(dur):,}  ({np.sum(dur == 0) / len(dur) * 100:.1f}%)")
        if len(nonzero) > 0:
            print(percentile_line(nonzero, "non-zero ", fmt_fn=fmt_duration))
            print()
            clip_hi = float(np.percentile(nonzero, 99))
            histogram_ascii(nonzero[nonzero <= clip_hi], bins=20, label="Non-zero duration (clipped at p99)")

    # ── Bytes per packet ─────────────────────────────────────────────────
    section("Bytes per Packet")
    bpp = d["bytes_per_packet"]
    if len(bpp) > 0:
        print(percentile_line(bpp, "bytes/pkt", fmt_fn=lambda x: f"{x:,.0f}"))
        print()
        clip_hi = float(np.percentile(bpp, 99))
        histogram_ascii(bpp[bpp <= clip_hi], bins=20, label="Bytes/packet (clipped at p99)")

    # ── Total bytes per flow ─────────────────────────────────────────────
    section("Total Bytes per Flow")
    tbf = d["total_bytes_per_flow"]
    if len(tbf) > 0:
        print(percentile_line(tbf, "bytes/flow", fmt_fn=lambda x: f"{x:,.0f}"))
        print()
        clip_hi = float(np.percentile(tbf, 99))
        histogram_ascii(tbf[tbf <= clip_hi], bins=20, label="Bytes/flow (clipped at p99)")

    # ── Inter-arrival times ──────────────────────────────────────────────
    section("Inter-Arrival Times (µs)")
    iats = d["iats_all"]
    if len(iats) > 0:
        print(f"  Total IAT values: {len(iats):,}")
        print(percentile_line(iats, "IAT     ", fmt_fn=fmt_duration))
        print()
        nonzero_iats = iats[iats > 0]
        print(f"  Zero IATs: {np.sum(iats == 0):,} / {len(iats):,}  ({np.sum(iats == 0) / len(iats) * 100:.1f}%)")
        if len(nonzero_iats) > 0:
            print(percentile_line(nonzero_iats, "non-zero", fmt_fn=fmt_duration))
            print()
            clip_hi = float(np.percentile(nonzero_iats, 99))
            histogram_ascii(nonzero_iats[nonzero_iats <= clip_hi], bins=20, label="Non-zero IAT (clipped at p99)")

    # ── Protocol breakdown ───────────────────────────────────────────────
    section("Protocol Distribution")
    proto_counts = Counter(d["protocols"])
    total_proto = sum(proto_counts.values())
    if total_proto > 0:
        for proto, cnt in sorted(proto_counts.items(), key=lambda x: -x[1]):
            name = PROTO_NAMES.get(proto, f"proto-{proto}")
            pct = cnt / total_proto * 100
            print(f"  {name:>8s} (={proto:>3d}) : {cnt:>10,} flows  ({pct:5.2f}%)")

    # ── Direction (fwd/bwd) ──────────────────────────────────────────────
    section("Direction Analysis (forward / backward packets)")
    fwd = d["fwd_counts"]
    bwd = d["bwd_counts"]
    if len(fwd) > 0:
        total_pkts = fwd.sum() + bwd.sum()
        print(f"  Total forward  packets: {fwd.sum():>12,}  ({fwd.sum() / total_pkts * 100:.1f}%)")
        print(f"  Total backward packets: {bwd.sum():>12,}  ({bwd.sum() / total_pkts * 100:.1f}%)")
        print()
        # Flows that are unidirectional
        uni_fwd = np.sum(bwd == 0)
        uni_bwd = np.sum(fwd == 0)
        bidir = np.sum((fwd > 0) & (bwd > 0))
        print(f"  Forward-only  flows: {uni_fwd:>10,}  ({uni_fwd / len(fwd) * 100:.1f}%)")
        print(f"  Backward-only flows: {uni_bwd:>10,}  ({uni_bwd / len(fwd) * 100:.1f}%)")
        print(f"  Bidirectional flows: {bidir:>10,}  ({bidir / len(fwd) * 100:.1f}%)")

    # ── Burst / token info ───────────────────────────────────────────────
    bl = d["burst_lengths"]
    if len(bl) > 0:
        section("Burst Tokens per Flow")
        print(percentile_line(bl, "bursts/flow", fmt_fn=lambda x: f"{x:.1f}"))
        print()
        vc = Counter(bl.tolist())
        print("  Top burst-count values:")
        for val, cnt in sorted(vc.items(), key=lambda x: -x[1])[:10]:
            pct = cnt / len(bl) * 100
            print(f"    {val:>6d} bursts : {cnt:>10,} flows  ({pct:5.2f}%)")

    # ── Counts per flow ──────────────────────────────────────────────────
    cpf = d["counts_per_flow"]
    if len(cpf) > 0:
        section("Counts Sum per Flow")
        print(percentile_line(cpf, "count_sum", fmt_fn=lambda x: f"{x:,.0f}"))

    print()
    print("=" * 80)
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Describe Arrow IPC dataset files (network flow data)."
    )
    parser.add_argument(
        "files", nargs="+", help="One or more .arrow files to describe"
    )
    args = parser.parse_args()

    for fpath in args.files:
        if not os.path.isfile(fpath):
            print(f"ERROR: File not found: {fpath}", file=sys.stderr)
            continue
        print(f"\nLoading {fpath} ...")
        data = load_arrow(fpath)
        report(fpath, data)


if __name__ == "__main__":
    main()
