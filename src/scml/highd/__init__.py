"""High-D track (coming soon): Graph-SCOPE + AdaGraph + DA-Sampler.

This track extends the SC-ML paradigm from low-dimensional point data to
high-dimensional / graph-structured data. The code is not yet released; the
methods are described in the AdaGraph preprint:

    https://arxiv.org/abs/2605.16320

The high-D track uses its own SLCD instantiation, Sample -> Learn -> Classify
-> Deploy, in which AdaGraph clusters a sample and the remaining points are
assigned to those clusters by voting. This differs from the low-D
Sample -> Label -> Calibrate -> Deploy, which transfers tuned parameters. The
two implementations are not interchangeable.
"""

COMING_SOON = True

__all__ = ["COMING_SOON"]
