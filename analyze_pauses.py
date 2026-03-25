"""Quick analysis of pause durations in pipeline results."""
import json
from collections import defaultdict
from pathlib import Path

results_path = Path("output/pipeline_results.json")
with open(results_path) as f:
    data = json.load(f)

# Handle both list and migrated formats
if isinstance(data, dict) and data.get("_migrated"):
    entries = list(data["data"].values())
else:
    entries = data if isinstance(data, list) else [data]

print("=" * 80)
print("PAUSE DURATION ANALYSIS")
print("=" * 80 + "\n")

all_pauses = []
pause_counts = defaultdict(int)

for entry in entries:
    if not isinstance(entry, dict):
        continue
    
    label = entry.get("label") or "unknown"
    pause_data = entry.get("agent_scores", {}).get("pause_patterns", {})
    
    if not pause_data:
        continue
    
    pauses = pause_data.get("pause_durations_ms", [])
    threshold = 200.0
    
    if pauses:
        below_threshold = sum(1 for p in pauses if p < threshold)
        above_threshold = sum(1 for p in pauses if p >= threshold)
        
        print(f"[{label}]")
        print(f"  Total pauses:     {len(pauses)}")
        print(f"  Below {threshold}ms:    {below_threshold}")
        print(f"  Above {threshold}ms:    {above_threshold}")
        print(f"  Mean duration:    {pause_data.get('mean_pause_ms', 0):.1f} ms")
        print(f"  Max duration:     {max(pauses):.1f} ms")
        print(f"  Min duration:     {min(pauses):.1f} ms")
        print(f"  Durations:        {[round(p, 1) for p in pauses[:10]]}")
        if len(pauses) > 10:
            print(f"                    ... and {len(pauses) - 10} more")
        print()
        
        all_pauses.extend(pauses)
        
        # Count by bucket
        for p in pauses:
            if p < 100:
                pause_counts["<100ms"] += 1
            elif p < 200:
                pause_counts["100-200ms"] += 1
            elif p < 500:
                pause_counts["200-500ms"] += 1
            elif p < 1000:
                pause_counts["500-1000ms"] += 1
            else:
                pause_counts[">1000ms"] += 1

print("\n" + "=" * 80)
print("AGGREGATE STATISTICS")
print("=" * 80 + "\n")

if all_pauses:
    import statistics
    print(f"Total pauses across all recordings: {len(all_pauses)}")
    print(f"Mean pause duration:                {statistics.mean(all_pauses):.1f} ms")
    print(f"Median pause duration:              {statistics.median(all_pauses):.1f} ms")
    print(f"Stdev:                              {statistics.stdev(all_pauses):.1f} ms")
    print(f"Min:                                {min(all_pauses):.1f} ms")
    print(f"Max:                                {max(all_pauses):.1f} ms")
    print()
    print("Distribution by threshold:")
    for bucket, count in sorted(pause_counts.items()):
        pct = 100 * count / len(all_pauses)
        print(f"  {bucket:12} : {count:3} ({pct:5.1f}%)")
    print()
    print("RECOMMENDATION:")
    count_below_150 = sum(1 for p in all_pauses if p < 150)
    count_below_200 = sum(1 for p in all_pauses if p < 200)
    pct_below_150 = 100 * count_below_150 / len(all_pauses)
    pct_below_200 = 100 * count_below_200 / len(all_pauses)
    print(f"  {pct_below_150:.1f}% of pauses are below 150ms")
    print(f"  {pct_below_200:.1f}% of pauses are below 200ms")
    
    if pct_below_150 > 50:
        print(f"  -> Consider lowering threshold from 200ms to 150ms")
    elif pct_below_200 > 70:
        print(f"  -> Consider lowering threshold from 200ms to 175ms")
else:
    print("No pauses found in results.")
