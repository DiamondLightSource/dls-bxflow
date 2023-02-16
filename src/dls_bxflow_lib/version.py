"""
The version of the package can be returned as a single string or a dict.

When a string, it comes from the package __version__.
When a dict, it also has __version__,
as well as versions of other depdency packages.
"""

# TODO: Use ImportlibMetadataVersion to get package versions.
# from importlib.metadata import PackageNotFoundError
# from importlib.metadata import version as ImportlibMetadataVersion

from typing import Optional

import dls_mainiac_lib.version
import dls_pairstream_lib.version
import dls_servbase_lib.version
import dls_utilpack.version

from dls_bxflow_lib import __version__


# ----------------------------------------------------------
def version() -> str:
    """
    Version of the dls_bxflow package as a string.
    """

    return __version__


# ----------------------------------------------------------
def meta(given_meta: Optional[dict] = None) -> dict:
    """
    Returns version information from the dls_bxflow package
    and its dependencies as a dict.
    Adds version information to a given meta dict if it was provided.
    """

    meta = {}
    meta["dls_bxflow"] = version()
    meta.update(dls_utilpack.version.meta())
    meta.update(dls_servbase_lib.version.meta())
    meta.update(dls_mainiac_lib.version.meta())
    meta.update(dls_utilpack.version.meta())
    meta.update(dls_pairstream_lib.version.meta())

    try:
        import stomp

        parts = stomp.__version__
        parts = [str(s) for s in parts]
        meta["stomp"] = ".".join(parts)
    except Exception:
        meta["stomp"] = "unavailable"

    try:
        import nbconvert

        meta["nbconvert"] = nbconvert.__version__
    except Exception:
        meta["nbconvert"] = "unavailable"

    try:
        import nbformat

        meta["nbformat"] = nbformat.__version__
    except Exception:
        meta["nbformat"] = "unavailable"

    try:
        import nbclient

        meta["nbclient"] = nbclient.__version__
    except Exception:
        meta["nbclient"] = "unavailable"

    try:
        import ispyb

        meta["ispyb"] = ispyb.__version__
    except Exception:
        meta["ispyb"] = "unavailable"

    if given_meta is not None:
        given_meta.update(meta)
    else:
        given_meta = meta
    return given_meta
