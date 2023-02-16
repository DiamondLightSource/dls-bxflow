from importlib.metadata import version

__version__ = version("dls-bxflow")
del version

__all__ = ["__version__"]
