"""
agent_diagnostics.py
--------------------
Per-agent diagnostic visualizer: shows score distributions across all recordings.

Detects misaligned calibration (e.g., pause_patterns stuck at 0.6–0.9)
and generates histograms, box plots, and summary statistics.

Usage:
    python agent_diagnostics.py
    python agent_diagnostics.py --output diagnostics/
    python agent_diagnostics.py --format json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class AgentDiagnosticsAnalyzer:
    """Analyse agent score distributions across pipeline results."""

    def __init__(self, results_path: str = "output/pipeline_results.json"):
        self.results_path = Path(results_path)
        self.agent_scores: Dict[str, List[float]] = defaultdict(list)
        self.agent_metadata: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "recordings": [],
        })
        self.raw_agent_data: Dict[str, List[Dict]] = defaultdict(list)  # Store full agent data per recording

    def load_results(self) -> None:
        """Load and parse pipeline results JSON."""
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_path}")

        with open(self.results_path) as f:
            data = json.load(f)

        # Handle both array format and migrated {"_migrated": true, "data": {...}} format
        if isinstance(data, dict) and data.get("_migrated"):
            entries = list(data["data"].values())
        else:
            entries = data

        if not isinstance(entries, list):
            entries = [entries]

        for entry in entries:
            if isinstance(entry, dict):
                # Try both old and new structures
                agent_scores = entry.get("agent_scores") or entry.get("data", {}).get("agent_scores")
                if not agent_scores:
                    continue

                label = entry.get("label") or "unknown"

                for agent_name, agent_result in agent_scores.items():
                    if isinstance(agent_result, dict) and "score" in agent_result:
                        score = agent_result["score"]
                        self.agent_scores[agent_name].append(score)
                        self.agent_metadata[agent_name]["count"] += 1
                        self.agent_metadata[agent_name]["recordings"].append(label)
                        
                        # Store full agent data for table generation
                        self.raw_agent_data[agent_name].append({
                            "label": label,
                            "score": score,
                            "data": agent_result,
                        })

    def compute_statistics(self) -> Dict[str, Dict]:
        """Compute per-agent statistics."""
        stats = {}
        for agent_name in sorted(self.agent_scores.keys()):
            scores = self.agent_scores[agent_name]
            if not scores:
                continue

            sorted_scores = sorted(scores)
            stats[agent_name] = {
                "count": len(scores),
                "min": round(min(scores), 3),
                "max": round(max(scores), 3),
                "mean": round(statistics.mean(scores), 3),
                "median": round(statistics.median(sorted_scores), 3),
                "stdev": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
                "q1": round(sorted_scores[len(scores) // 4], 3),
                "q3": round(sorted_scores[3 * len(scores) // 4], 3),
                "iqr": round(
                    sorted_scores[3 * len(scores) // 4] - sorted_scores[len(scores) // 4],
                    3,
                ),
            }

            # Detect bunching (poor calibration signal)
            iqr = stats[agent_name]["iqr"]
            spread = stats[agent_name]["max"] - stats[agent_name]["min"]
            if spread < 0.3 and len(scores) > 3:
                stats[agent_name]["_warning"] = (
                    f"Low spread ({spread:.2f}) — scores bunched between "
                    f"{stats[agent_name]['min']} and {stats[agent_name]['max']}. "
                    f"Calibration ceiling may be too low."
                )

        return stats

    def print_summary(self, stats: Dict[str, Dict]) -> None:
        """Print human-readable summary statistics."""
        print("\n" + "=" * 80)
        print("AGENT DIAGNOSTICS SUMMARY".center(80))
        print("=" * 80 + "\n")

        for agent_name in sorted(stats.keys()):
            s = stats[agent_name]
            print(f"[{agent_name.upper()}]")
            print(f"  Recordings: {s['count']}")
            print(f"  Range:      {s['min']} to {s['max']}  (spread: {s['max'] - s['min']:.3f})")
            print(f"  Mean:       {s['mean']}")
            print(f"  Median:     {s['median']}")
            print(f"  Stdev:      {s['stdev']}")
            print(f"  IQR:        {s['q1']} to {s['q3']}  (width: {s['iqr']:.3f})")

            if "_warning" in s:
                print(f"  [WARNING] {s['_warning']}")

            print()

    def generate_agent_tables(self) -> str:
        """Generate detailed text tables for each agent with per-recording metrics."""
        output = []
        output.append("\n" + "=" * 100)
        output.append("DETAILED AGENT METRICS PER RECORDING".center(100))
        output.append("=" * 100 + "\n")

        # Speech Rate Agent Table
        if "speech_rate" in self.raw_agent_data:
            output.append("\n" + "-" * 100)
            output.append("SPEECH RATE AGENT".ljust(100))
            output.append("-" * 100)
            output.append(f"{'Recording':<15} {'WPM':<8} {'Variance':<12} {'Slow Score':<12} {'Rush Score':<12} {'Final Score':<12}")
            output.append("-" * 100)
            
            for entry in self.raw_agent_data["speech_rate"]:
                data = entry["data"]
                output.append(
                    f"{entry['label']:<15} "
                    f"{data.get('wpm', 0):<8.1f} "
                    f"{data.get('wpm_variance', 0):<12.2f} "
                    f"{data.get('slow_score', 0):<12.3f} "
                    f"{data.get('rush_score', 0):<12.3f} "
                    f"{data.get('score', 0):<12.3f}"
                )
            output.append("-" * 100)
            output.append("Notes: Measures words-per-minute, variance, slowness, and speech rushing.")
            output.append("       Higher WPM variance and slow/rush scores indicate cognitive load.\n")

        # Pause Patterns Agent Table
        if "pause_patterns" in self.raw_agent_data:
            output.append("\n" + "-" * 100)
            output.append("PAUSE PATTERNS AGENT".ljust(100))
            output.append("-" * 100)
            output.append(
                f"{'Recording':<15} {'Pauses':<8} {'Mean(ms)':<12} {'Rate/min':<12} {'Long%':<10} {'Final Score':<12}"
            )
            output.append("-" * 100)
            
            for entry in self.raw_agent_data["pause_patterns"]:
                data = entry["data"]
                output.append(
                    f"{entry['label']:<15} "
                    f"{data.get('pause_count', 0):<8} "
                    f"{data.get('mean_pause_ms', 0):<12.1f} "
                    f"{data.get('pause_rate_per_min', 0):<12.2f} "
                    f"{data.get('long_pause_fraction', 0):<10.1%} "
                    f"{data.get('score', 0):<12.3f}"
                )
            output.append("-" * 100)
            output.append("Notes: Detects hesitation: count, mean duration, rate, and long pauses (>1200ms).")
            output.append("       Higher pause frequency/duration = higher cognitive load.\n")

        # Filler Words Agent Table
        if "filler_words" in self.raw_agent_data:
            output.append("\n" + "-" * 100)
            output.append("FILLER WORDS AGENT".ljust(100))
            output.append("-" * 100)
            output.append(
                f"{'Recording':<15} {'Total Words':<13} {'Fillers':<8} {'Rate':<10} {'Weighted':<10} {'Final Score':<12}"
            )
            output.append("-" * 100)
            
            for entry in self.raw_agent_data["filler_words"]:
                data = entry["data"]
                output.append(
                    f"{entry['label']:<15} "
                    f"{data.get('total_words', 0):<13} "
                    f"{data.get('total_fillers', 0):<8} "
                    f"{data.get('filler_rate', 0):<10.1%} "
                    f"{data.get('weighted_rate', 0):<10.1%} "
                    f"{data.get('score', 0):<12.3f}"
                )
            output.append("-" * 100)
            output.append("Notes: Detects disfluency markers (um, uh, like, so, etc.).")
            output.append("       'Um'/'uh' weighted 2x; others weighted 1x.\n")

        # Semantic Density Agent Table
        if "semantic_density" in self.raw_agent_data:
            output.append("\n" + "-" * 100)
            output.append("SEMANTIC DENSITY AGENT".ljust(100))
            output.append("-" * 100)
            output.append(
                f"{'Recording':<15} {'Density Score':<15} {'Final Score':<12} {'Truncated':<12} {'Reasoning':<40}"
            )
            output.append("-" * 100)
            
            for entry in self.raw_agent_data["semantic_density"]:
                data = entry["data"]
                reasoning = data.get('reasoning', '')[:37] + '...' if len(data.get('reasoning', '')) > 40 else data.get('reasoning', '')
                output.append(
                    f"{entry['label']:<15} "
                    f"{data.get('density_score', 0):<15.1f} "
                    f"{data.get('score', 0):<12.3f} "
                    f"{'Yes' if data.get('truncated') else 'No':<12} "
                    f"{reasoning:<40}"
                )
            output.append("-" * 100)
            output.append("Notes: LLM-based information density measurement (0=high info, 1=low info).")
            output.append("       Score inverted: Low density → high cognitive load.\n")

        # Syntactic Complexity Agent Table
        if "syntactic_complexity" in self.raw_agent_data:
            output.append("\n" + "-" * 100)
            output.append("SYNTACTIC COMPLEXITY AGENT".ljust(100))
            output.append("-" * 100)
            output.append(
                f"{'Recording':<15} {'Sentences':<12} {'Avg Len':<10} {'Depth':<8} {'Sub%':<8} {'Unreliable':<12} {'Final Score':<12}"
            )
            output.append("-" * 100)
            
            for entry in self.raw_agent_data["syntactic_complexity"]:
                data = entry["data"]
                unreliable_str = "YES" if data.get('unreliable') else "NO"
                output.append(
                    f"{entry['label']:<15} "
                    f"{data.get('sentence_count', 0):<12} "
                    f"{data.get('avg_sentence_len', 0):<10.1f} "
                    f"{data.get('avg_tree_depth', 0):<8.1f} "
                    f"{data.get('subordination_rate', 0):<8.1%} "
                    f"{unreliable_str:<12} "
                    f"{data.get('score', 0):<12.3f}"
                )
            output.append("-" * 100)
            output.append("Notes: Dependency parse analysis of sentence structure complexity.")
            output.append("       Marked UNRELIABLE if <3 sentences (fallback to 0.5 score).\n")

        return "\n".join(output)

    def export_json(self, output_path: Path, stats: Dict[str, Dict]) -> None:
        """Export statistics to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "summary": stats,
                    "by_agent": {
                        agent: list(self.agent_scores[agent])
                        for agent in sorted(self.agent_scores.keys())
                    },
                },
                f,
                indent=2,
            )
        print(f"Exported JSON → {output_path}")

    def export_tables(self, output_dir: Path) -> None:
        """Export agent summary tables to text file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        tables_text = self.generate_agent_tables()
        
        output_path = output_dir / "agent_summary_tables.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(tables_text)
        
        print(f"Exported tables → {output_path}")

    def plot_distributions(self, output_dir: Path = None) -> None:
        """Create histograms and box plots for each agent."""
        if not MATPLOTLIB_AVAILABLE:
            print("[WARNING] matplotlib not available — skipping plots.")
            return

        output_dir = output_dir or Path("diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Overall figure: all agents in one view
        agents = sorted(self.agent_scores.keys())
        n_agents = len(agents)

        fig, axes = plt.subplots(n_agents, 2, figsize=(14, 4 * n_agents))
        if n_agents == 1:
            axes = axes.reshape(1, -1)

        for idx, agent_name in enumerate(agents):
            scores = self.agent_scores[agent_name]

            # Histogram
            ax_hist = axes[idx, 0]
            ax_hist.hist(scores, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
            ax_hist.axvline(statistics.mean(scores), color="red", linestyle="--", label="Mean")
            ax_hist.axvline(statistics.median(sorted(scores)), color="green", linestyle="--", label="Median")
            ax_hist.set_title(f"{agent_name} — Score Distribution")
            ax_hist.set_xlabel("Score")
            ax_hist.set_ylabel("Frequency")
            ax_hist.legend()
            ax_hist.grid(alpha=0.3)

            # Box plot
            ax_box = axes[idx, 1]
            ax_box.boxplot(scores, vert=True)
            ax_box.set_title(f"{agent_name} — Box Plot")
            ax_box.set_ylabel("Score")
            ax_box.grid(alpha=0.3, axis="y")

            # Highlight spread
            spread = max(scores) - min(scores)
            color = "red" if spread < 0.3 else "green"
            ax_box.set_facecolor("pink" if spread < 0.3 else "lightgreen")

        plt.tight_layout()
        plot_path = output_dir / "agent_distributions.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {plot_path}")
        plt.close()

        # Per-agent detailed histograms
        for agent_name in agents:
            scores = self.agent_scores[agent_name]
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.hist(scores, bins=12, color="royalblue", edgecolor="black", alpha=0.7)
            ax.axvline(statistics.mean(scores), color="red", linestyle="--", linewidth=2, label="Mean")
            ax.axvline(statistics.median(sorted(scores)), color="green", linestyle="--", linewidth=2, label="Median")

            # Shade IQR
            sorted_scores = sorted(scores)
            q1 = sorted_scores[len(scores) // 4]
            q3 = sorted_scores[3 * len(scores) // 4]
            ax.axvspan(q1, q3, alpha=0.2, color="orange", label="IQR")

            ax.set_title(f"{agent_name} — Detailed Distribution (n={len(scores)})")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

            plot_path = output_dir / f"{agent_name}_histogram.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"Saved per-agent histograms to {output_dir}/")

    def run(self, output_dir: str = None, export_json: bool = False) -> None:
        """Run full diagnostic analysis."""
        print("Loading results...")
        self.load_results()

        if not self.agent_scores:
            print("❌ No agent scores found in results file.")
            return

        print(f"[OK] Loaded {sum(len(v) for v in self.agent_scores.values())} scores across {len(self.agent_scores)} agents\n")

        stats = self.compute_statistics()
        self.print_summary(stats)

        # Generate and print agent tables
        tables_text = self.generate_agent_tables()
        print(tables_text)

        # Visualizations
        output_dir = Path(output_dir) if output_dir else Path("diagnostics")
        self.plot_distributions(output_dir)

        # Export tables and JSON
        self.export_tables(output_dir)
        if export_json:
            self.export_json(output_dir / "agent_diagnostics.json", stats)


def main():
    parser = argparse.ArgumentParser(
        description="Analyse per-agent score distributions in pipeline results."
    )
    parser.add_argument(
        "-i", "--input",
        default="output/pipeline_results.json",
        help="Path to pipeline results JSON (default: output/pipeline_results.json)",
    )
    parser.add_argument(
        "-o", "--output",
        default="diagnostics",
        help="Output directory for plots (default: diagnostics/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export statistics to JSON as well",
    )

    args = parser.parse_args()

    analyzer = AgentDiagnosticsAnalyzer(args.input)
    analyzer.run(output_dir=args.output, export_json=args.json)


if __name__ == "__main__":
    main()
