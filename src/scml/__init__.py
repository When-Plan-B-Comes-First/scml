"""scml: Structure-Centric Machine Learning.

A unified package for the SC-ML paradigm by Dr. Ahmed Elmahdi.

Two tracks:
  * Low-D  (shipping now):    SCOPE + AdaBox + SLCD   -> scml.lowd
  * High-D (live):            AdaGraph + Graph-SCOPE + SLCD -> scml.highd

Quickstart:
    from scml.lowd import AdaBox, scope_score
"""

__version__ = "0.2.0"

from . import lowd
from . import highd

__all__ = ["lowd", "highd", "__version__"]
