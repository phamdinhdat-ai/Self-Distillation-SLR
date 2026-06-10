"""
Word Error Rate (WER) metric for CSLR evaluation.

WER = (substitutions + insertions + deletions) / len(reference)
computed via edit-distance (Wagner-Fischer dynamic programming).
"""

from typing import List


def edit_distance(hyp: List[int], ref: List[int]) -> int:
    """Levenshtein edit distance between two sequences."""
    H, R = len(hyp), len(ref)
    # DP table
    dp = list(range(R + 1))
    for i in range(1, H + 1):
        new_dp = [i] + [0] * R
        for j in range(1, R + 1):
            if hyp[i - 1] == ref[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j],       # deletion
                                    new_dp[j - 1], # insertion
                                    dp[j - 1])     # substitution
        dp = new_dp
    return dp[R]


def compute_wer(
    hypotheses: List[List[int]],
    references: List[List[int]],
) -> float:
    """
    Compute corpus-level WER.

    Args:
        hypotheses: list of decoded token-id sequences
        references: list of reference token-id sequences
    Returns:
        WER as a percentage (0–100)
    """
    total_dist = 0
    total_len  = 0
    for hyp, ref in zip(hypotheses, references):
        total_dist += edit_distance(hyp, ref)
        total_len  += len(ref)
    if total_len == 0:
        return 0.0
    return 100.0 * total_dist / total_len
