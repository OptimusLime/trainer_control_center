"""Model classes for ACC experiments.

All models return dict[ModelOutput, Tensor] from forward().
The Trainer is model-agnostic — any nn.Module conforming to
this protocol works.
"""
