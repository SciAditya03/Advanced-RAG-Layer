"""Context compression and prompt construction."""

__all__ = ["compress_for_officer", "build_teaching_prompt"]

from .compressor import compress_for_officer
from .prompt_builder import build_teaching_prompt