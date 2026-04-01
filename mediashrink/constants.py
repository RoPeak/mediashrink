from __future__ import annotations

# Approximate output-size ratio relative to source for libx265 at CRF 20.
# Used by wizard profiles for display estimates and by encoder for file-size estimation.
# Keys are CRF values; values are the fraction of input bytes expected in the output.
CRF_COMPRESSION_FACTOR: dict[int, float] = {
    16: 0.60,
    18: 0.50,
    20: 0.40,
    22: 0.34,
    24: 0.28,
    28: 0.22,
}

# CRF value used as the reference baseline when scaling codec/resolution factors.
CRF_BASELINE = 20
