# Copyright © 2023-2025 Apple Inc.
"""
MergeKit for MLX - Advanced model merging functionality for MLX-LM
"""

from .merge_advanced import merge_advanced

# Version info
__version__ = '0.1.0'  # Initial release version

# Export the main function as a simpler alias
mergekit = merge_advanced

# Export all functions
__all__ = ["merge_advanced", "mergekit"]
