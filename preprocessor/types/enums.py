"""Enums shared by request/response models."""

from enum import Enum


class ResultStatus(str, Enum):
    """Standardized status for hand and motion results."""

    OK = "ok"
    NO_HAND = "no_hand"
    ERROR = "error"
