# Graph-SCOPE (coming soon)

Graph-SCOPE is the high-dimensional / graph-structured extension of the SCOPE
metric. Where SCOPE evaluates clustering structure on low-D point data,
Graph-SCOPE generalises the same five-component philosophy (core purity,
boundary recall, cluster precision, noise handling, count accuracy) to graph
and high-D embeddings.

**Status:** released as a preprint; reference implementation is being prepared
for this repository.

**Preprint:** Graph-SCOPE is introduced alongside the high-D track; see the
AdaGraph preprint — <https://arxiv.org/abs/2605.16320>

The low-D SCOPE is available now: `from scml.lowd import scope_score`.
