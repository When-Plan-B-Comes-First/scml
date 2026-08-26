"""scml: Structure-Centric Machine Learning.

A unified package for the SC-ML paradigm by Dr. Ahmed Elmahdi.

Two tracks:
  * Low-D  (shipping now):    SCOPE + AdaBox + SLCD   -> scml.lowd
  * High-D (coming soon):     Graph-SCOPE + AdaGraph + DA-Sampler -> scml.highd

Quickstart:
    from scml.lowd import AdaBox, scope_score
"""

__version__ = "0.1.0"

from . import lowd

__all__ = ["lowd", "__version__"]
