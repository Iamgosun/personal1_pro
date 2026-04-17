from __future__ import annotations

# 这里不重复手写底层 token 逻辑，而是显式复用 legacy 实现。


def get_legacy_modules():
    from trainers.mmrl import MultiModalRepresentationLearner
    return {'SharedRepresentationTokens': MultiModalRepresentationLearner}
