"""A minimal tool-using agent: a fixed causal toolset plus a model-agnostic tool-calling loop.

``tools.py`` defines the safe, parameterized causal-analysis functions (no arbitrary code execution);
``loop.py`` runs a provider-agnostic tool-calling loop against them; ``eval.py`` batches the loop over
many instances and aggregates scores by naming alignment.
"""
