"""Command engine for gesture-to-media-control mapping (Module E).

This package translates classifier predictions into keyboard actions
with safety logic (debounce, cooldown, confidence thresholds) to
prevent accidental or repeated triggers with varying framerates.

Integrates with the preprocessor's ResultStatus to handle NO_HAND
frames as natural reset points for debounce and gesture state.
"""
