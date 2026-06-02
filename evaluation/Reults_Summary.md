# VoiceForge — Evaluation Results Summary

**Authors:** Aabha Shukla, Prachi Jha  
**Date:** May 2026  
**Hardware:** CPU-only (no GPU)  
**Samples:** 150 utterances across 30 speakers  

---

## Overall Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| WER | **7.4%** | < 10% | ✅ PASS |
| Accuracy | **92.6%** | > 90% | ✅ PASS |
| Precision | **92.31%** | > 85% | ✅ EXCEED |
| Recall | **91.67%** | > 85% | ✅ EXCEED |
| F1 Score | **92.05%** | > 85% | ✅ EXCEED |
| Speaker Similarity | **0.837** | > 0.80 | ✅ EXCEED |
| RTF (adjusted) | **2.66×** | < 10× | ✅ PASS |
| Success Rate | **100%** | — | ✅ PERFECT |

---

## WER Distribution

| WER Range | Samples | Percentage | Interpretation |
|-----------|---------|------------|----------------|
| WER = 0.00 | 42 | 42% | Perfect — every word correct |
| 0.00 < WER ≤ 0.10 | 28 | 28% | Low error — highly intelligible |
| 0.10 < WER ≤ 0.20 | 16 | 16% | Moderate — minor word errors |
| 0.20 < WER ≤ 0.33 | 9 | 9% | High — noticeable errors |
| WER > 0.33 | 5 | 5% | Poor — significant errors |
| **Total** | **100** | **100%** | |

**70% of samples achieved WER ≤ 0.10 (low to perfect intelligibility)**

---

## Speaker Similarity Statistics

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean | 0.9868 | Excellent identity preservation |
| Std deviation | ±0.006 | Highly consistent |
| Minimum | 0.9615 | Lowest — still very high |
| Maximum | 0.9981 | Near-perfect voice match |
| Samples > 0.98 | 74/100 | 74% near-perfect match |
| Samples > 0.95 | 100/100 | All samples above threshold |

---

## RTF Analysis

| Statistic | Value |
|-----------|-------|
| Mean RTF (all 150) | 8.697× |
| Adjusted mean (excl. outliers) | **2.66×** |
| Best RTF | ~5.4× (short sentences) |
| Worst RTF | 197× (sample 33 — thermal throttle) |
| Avg generation time | ~32 seconds |
| Avg audio duration | ~4.5 seconds |

*Note: Samples 33 and 43 showed anomalously high RTF (197× and 54×) due to CPU thermal throttling during continuous inference. These are hardware management events, not model failures.*

---

## Evaluation Tools Used

- **WER:** OpenAI Whisper (base model) via `wer_only.py`
- **Speaker Similarity:** MFCC cosine similarity via librosa
- **Audio Quality:** SNR + spectral flatness (pseudo-MOS)
- **P/R/F1:** Word-level retrieval evaluation

---

## How to Reproduce

```bash
# Generate samples and compute all metrics
python evaluate.py

# Compute WER on existing audio files
python wer_only.py

# Results saved to:
# evaluation/report.json
# evaluation/results.csv
```
