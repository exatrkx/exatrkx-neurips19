"""
GNN models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_model(model_name=None):
    from .model          import SegmentClassifier as mm
    from .model_more     import SegmentClassifier as mm_more
    from .model_smart    import SegmentClassifier as mm_sm
    from .model_vary     import SegmentClassifier as mm_vary
    from .model_vary2    import SegmentClassifier as mm_vary2

    # model_name could be used for future testing different models

    if model_name == "MORE":
        return mm_more()
    elif model_name == "SMART":
        return mm_sm()
    elif model_name == "VARY":
        return mm_vary()
    elif model_name == "VARY2":
        return mm_vary2()
    else:
        return mm()

