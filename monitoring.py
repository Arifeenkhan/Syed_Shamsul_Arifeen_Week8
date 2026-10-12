import re
import numpy as np
from scipy.stats import ks_2samp


def text_statistics(text: str) -> dict:
    words = re.findall(r"\b\w+\b", text.lower())
    return {
        "char_count": len(text),
        "word_count": len(words),
        "mean_word_length": float(
            np.mean([len(w) for w in words]) if words else 0.0
        ),
        "uppercase_ratio": (
            sum(c.isupper() for c in text) / len(text) if text else 0.0
        ),
        "punctuation_ratio": (
            sum(not c.isalnum() and not c.isspace() for c in text)
            / len(text) if text else 0.0
        ),
    }


class DriftMonitor:
    def __init__(self, reference_stats: list[dict], threshold: float = 0.05):
        if len(reference_stats) < 20:
            raise ValueError("Use at least 20 reference comments.")
        self.reference_stats = reference_stats
        self.threshold = threshold
        self.recent_stats = []

    def observe(self, text: str) -> dict:
        stats = text_statistics(text)
        self.recent_stats.append(stats)

        if len(self.recent_stats) < 30:
            return {
                "drift_detected": False,
                "status": "collecting_baseline_window",
                "window_size": len(self.recent_stats),
                "statistics": stats,
            }

        features = [
            "char_count", "word_count", "mean_word_length",
            "uppercase_ratio", "punctuation_ratio"
        ]
        p_values = {}

        for feature in features:
            reference = [s[feature] for s in self.reference_stats]
            current = [s[feature] for s in self.recent_stats[-100:]]
            p_values[feature] = float(
                ks_2samp(reference, current).pvalue
            )

        cutoff = self.threshold / len(features)
        flagged = [f for f, p in p_values.items() if p < cutoff]

        return {
            "drift_detected": bool(flagged),
            "status": "drift_detected" if flagged else "no_drift_detected",
            "flagged_features": flagged,
            "p_values": p_values,
            "window_size": min(len(self.recent_stats), 100),
            "statistics": stats,
        }
