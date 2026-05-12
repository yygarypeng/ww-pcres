from importlib import import_module

__all__ = [
    "clean_training_output",
    "flatten_config",
    "load_config",
    "main",
    "prime_csv_metric_header",
]


def __getattr__(name):
    if name in __all__:
        return getattr(import_module("train.main"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
