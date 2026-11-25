"""Offline multilingual localization layer (`en`, `hi`, `gu`, `mr`) for FieldWise PS1."""

from .translations import (
    SUPPORTED_LANGUAGES,
    get_ui_bundle,
    localize_condition_name,
    localize_crop_name,
    localize_status_message,
)

__all__ = [
    "SUPPORTED_LANGUAGES",
    "get_ui_bundle",
    "localize_condition_name",
    "localize_crop_name",
    "localize_status_message",
]
