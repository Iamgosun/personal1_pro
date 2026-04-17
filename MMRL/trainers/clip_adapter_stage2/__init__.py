"""Stage-2 refactor modules for clip_adapters trainer.

This package keeps the external trainer name `ClipADAPTER` unchanged,
but moves text-feature extraction, adapter variants, feature caching,
and result reporting into smaller modules.
"""

from .text import CUSTOM_TEMPLATES, TextEncoder, get_base_text_features, load_clip_to_cpu
from .adapter_variants import AdapterMethod, build_adapter
from .model import CustomCLIP
from .cache import FeatureCacheManager
from .feature_pipeline import FeaturePipeline
from .reporting import save_final_reports, save_confidence_coverage_stats

__all__ = [
    "CUSTOM_TEMPLATES",
    "TextEncoder",
    "get_base_text_features",
    "load_clip_to_cpu",
    "AdapterMethod",
    "build_adapter",
    "CustomCLIP",
    "FeatureCacheManager",
    "FeaturePipeline",
    "save_final_reports",
    "save_confidence_coverage_stats",
]
