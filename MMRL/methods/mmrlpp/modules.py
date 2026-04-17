from __future__ import annotations


def get_legacy_modules():
    from trainers.mmrlpp import Residual_Aligner, Shared_Residual_Representation_Aligner
    return {
        'ResidualAligner': Residual_Aligner,
        'SharedResidualRepresentationAligner': Shared_Residual_Representation_Aligner,
    }
