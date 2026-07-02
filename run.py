#!/usr/bin/env python3
"""
NeuroIgniter AI Recruiter v2 — Quick Run
========================================
python run.py [input.jsonl[.gz]] [output.csv]
"""
import os
import sys
import csv
import time

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))

    print("=" * 62)
    print("  NeuroIgniter AI Recruiter v2")
    print("  Team: NeuroIgniter | India Runs Data & AI Challenge")
    print("=" * 62)

    data_dir = os.path.join(repo_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(repo_root, 'output'), exist_ok=True)

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    elif os.path.exists(os.path.join(data_dir, 'candidates.jsonl.gz')):
        input_path = os.path.join(data_dir, 'candidates.jsonl.gz')
    elif os.path.exists(os.path.join(data_dir, 'candidates.jsonl')):
        input_path = os.path.join(data_dir, 'candidates.jsonl')
    else:
        print("ERROR: Place candidates.jsonl or candidates.jsonl.gz in data/")
        sys.exit(1)

    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(repo_root, 'output', 'neuroigniter_submission.csv')
    stats_path = os.path.join(repo_root, 'output', 'ranking_stats.json')

    # Add src/ to path
    sys.path.insert(0, os.path.join(repo_root, 'src'))

    from neuroigniter.ranker import rank_candidates

    t0 = time.time()
    top_results = rank_candidates(input_path, output_path, stats_output_path=stats_path)
    elapsed = time.time() - t0

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Output: {output_path}")

    # Validate
    with open(output_path, 'r') as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    scores = [float(r['score']) for r in rows]
    monotonic = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    unique_ids = len(set(r['candidate_id'] for r in rows))

    print(f"\n  Validation:")
    print(f"    Rows: {n} {'✓' if n == 100 else '✗ PROBLEM'}")
    print(f"    Scores monotonic: {'✓' if monotonic else '✗ PROBLEM'}")
    print(f"    Unique IDs: {unique_ids} {'✓' if unique_ids == 100 else '✗ PROBLEM'}")
    print(f"    Score range: {scores[-1]:.4f} – {scores[0]:.4f}")

    if n == 100 and monotonic and unique_ids == 100:
        print("\n  ✅ Submission valid — ready to submit!")
    else:
        print("\n  ❌ Validation errors — check output")

    print("\n  Top 10:")
    for r in rows[:10]:
        print(f"    #{r['rank']:>3}: {r['candidate_id']}  {float(r['score']):.4f}")
        print(f"         {r['reasoning'][:100]}...")


if __name__ == '__main__':
    main()
