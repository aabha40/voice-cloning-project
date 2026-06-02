"""
wer_analysis.py — Detailed WER Analysis & Visualisation
Author: Prachi Jha (prachijhaa.2901@gmail.com)
CDAC Pune Internship — VoiceForge Project

This script reads the evaluation CSV and produces:
1. WER distribution breakdown
2. Category-wise analysis (sentence type)
3. Precision/Recall/F1 summary
4. Comparison against industry thresholds
5. Console report + saves analysis to evaluation/analysis_report.txt
"""

import csv
import os
import json
import numpy as np
from collections import defaultdict

REPORT_JSON = "evaluation/report.json"
CSV_PATH = "evaluation/results.csv"
OUTPUT_TXT = "evaluation/analysis_report.txt"

# Industry benchmarks
BENCHMARKS = {
    "wer": {"threshold": 0.10, "direction": "lower", "label": "WER < 10%"},
    "accuracy": {"threshold": 0.90, "direction": "higher", "label": "Accuracy > 90%"},
    "precision": {"threshold": 0.85, "direction": "higher", "label": "Precision > 85%"},
    "recall": {"threshold": 0.85, "direction": "higher", "label": "Recall > 85%"},
    "f1": {"threshold": 0.85, "direction": "higher", "label": "F1 > 85%"},
    "speaker_sim": {"threshold": 0.80, "direction": "higher", "label": "SIM > 0.80"},
}

def load_results():
    """Load evaluation results from JSON report."""
    if not os.path.exists(REPORT_JSON):
        print(f"⚠ Report not found at {REPORT_JSON}")
        print("  Run evaluate.py or wer_only.py first.")
        return None
    with open(REPORT_JSON) as f:
        return json.load(f)

def wer_distribution(samples):
    """Compute WER distribution across buckets."""
    buckets = {
        "WER = 0.00 (Perfect)": 0,
        "0.00 < WER ≤ 0.10 (Low)": 0,
        "0.10 < WER ≤ 0.20 (Moderate)": 0,
        "0.20 < WER ≤ 0.33 (High)": 0,
        "WER > 0.33 (Poor)": 0,
    }
    wer_values = []
    for s in samples:
        w = s.get("wer")
        if w is None:
            continue
        wer_values.append(w)
        if w == 0.0:
            buckets["WER = 0.00 (Perfect)"] += 1
        elif w <= 0.10:
            buckets["0.00 < WER ≤ 0.10 (Low)"] += 1
        elif w <= 0.20:
            buckets["0.10 < WER ≤ 0.20 (Moderate)"] += 1
        elif w <= 0.33:
            buckets["0.20 < WER ≤ 0.33 (High)"] += 1
        else:
            buckets["WER > 0.33 (Poor)"] += 1
    return buckets, wer_values

def check_benchmarks(metrics):
    """Check all metrics against industry thresholds."""
    results = {}
    wer_mean = metrics.get("wer", {}).get("mean")
    if isinstance(wer_mean, float):
        results["wer"] = {"value": wer_mean, "pass": wer_mean < 0.10}
        results["accuracy"] = {"value": 1 - wer_mean, "pass": (1 - wer_mean) > 0.90}
    results["speaker_sim"] = {
        "value": metrics.get("speaker_similarity", {}).get("mean", 0),
        "pass": metrics.get("speaker_similarity", {}).get("mean", 0) > 0.80
    }
    results["precision"] = {
        "value": metrics.get("precision", {}).get("mean", 0),
        "pass": (metrics.get("precision", {}).get("mean", 0) or 0) > 0.85
    }
    results["recall"] = {
        "value": metrics.get("recall", {}).get("mean", 0),
        "pass": (metrics.get("recall", {}).get("mean", 0) or 0) > 0.85
    }
    results["f1"] = {
        "value": metrics.get("f1_score", {}).get("mean", 0),
        "pass": (metrics.get("f1_score", {}).get("mean", 0) or 0) > 0.85
    }
    return results

def print_banner(title):
    width = 60
    print("=" * width)
    print(f"  {title}")
    print("=" * width)

def generate_report(data):
    """Generate and print the analysis report."""
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 60)
    log("  VoiceForge — Evaluation Analysis Report")
    log(f"  Authors: Aabha Shukla, Prachi Jha | CDAC Pune")
    log("=" * 60)

    m = data.get("metrics", {})
    samples = data.get("per_sample", [])
    valid = [s for s in samples if "error" not in s]

    log(f"\n  Total samples   : {data.get('total_samples')}")
    log(f"  Successful      : {data.get('successful')}")
    log(f"  Failed          : {data.get('failed')}")
    log(f"  Reference audio : {data.get('reference_wav')}")
    log(f"  Device          : {data.get('device')}")

    log("\n── SPEAKER SIMILARITY ─────────────────────────────────")
    sim = m.get("speaker_similarity", {})
    log(f"  Mean   : {sim.get('mean', 'N/A'):.4f}")
    log(f"  Std    : ±{sim.get('std', 0):.4f}")
    log(f"  Min    : {sim.get('min', 'N/A'):.4f}")
    log(f"  Max    : {sim.get('max', 'N/A'):.4f}")
    above_98 = sum(1 for s in valid if s.get("speaker_sim", 0) >= 0.98)
    above_95 = sum(1 for s in valid if s.get("speaker_sim", 0) >= 0.95)
    log(f"  Above 0.98 : {above_98}/{len(valid)}")
    log(f"  Above 0.95 : {above_95}/{len(valid)}")

    log("\n── WER DISTRIBUTION ───────────────────────────────────")
    buckets, wer_vals = wer_distribution(valid)
    total_wer = len(wer_vals) or 1
    for bucket, count in buckets.items():
        pct = count / total_wer * 100
        bar = "█" * int(pct / 3)
        log(f"  {bucket:<35} {count:3d} ({pct:5.1f}%) {bar}")
    if wer_vals:
        log(f"\n  Mean WER   : {np.mean(wer_vals):.4f}  ({(1-np.mean(wer_vals))*100:.1f}% accuracy)")
        log(f"  Std WER    : ±{np.std(wer_vals):.4f}")

    log("\n── RTF ANALYSIS ───────────────────────────────────────")
    rtf_vals = [s.get("rtf", 0) for s in valid if s.get("rtf")]
    if rtf_vals:
        rtf_arr = np.array(rtf_vals)
        log(f"  Mean RTF (all)   : {np.mean(rtf_arr):.3f}×")
        outlier_thresh = np.percentile(rtf_arr, 95)
        filtered = rtf_arr[rtf_arr < outlier_thresh]
        log(f"  Mean RTF (adj.)  : {np.mean(filtered):.3f}× (excl. top 5%)")
        log(f"  Best RTF         : {np.min(rtf_arr):.3f}×")
        log(f"  Worst RTF        : {np.max(rtf_arr):.3f}×")

    log("\n── BENCHMARK COMPARISON ───────────────────────────────")
    bench = check_benchmarks(m)
    for key, val in bench.items():
        status = "✅ PASS" if val.get("pass") else "❌ FAIL"
        v = val.get("value")
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        log(f"  {key:<20} {v_str:<12} {status}")

    log("\n" + "=" * 60)
    log("  Analysis by: Prachi Jha (prachijhaa.2901@gmail.com)")
    log("=" * 60)

    return "\n".join(lines)

def main():
    print("VoiceForge — WER Analysis Tool")
    print("Author: Prachi Jha | CDAC Pune\n")

    data = load_results()
    if not data:
        return

    report = generate_report(data)

    os.makedirs("evaluation", exist_ok=True)
    with open(OUTPUT_TXT, "w") as f:
        f.write(report)
    print(f"\n✓ Analysis saved to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
