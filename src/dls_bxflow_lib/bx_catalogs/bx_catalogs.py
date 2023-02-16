# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_bx_catalog = None


def bx_catalogs_set_default(bx_catalog):
    global __default_bx_catalog
    __default_bx_catalog = bx_catalog


def bx_catalogs_get_default():
    global __default_bx_catalog
    if __default_bx_catalog is None:
        raise RuntimeError("bx_catalogs_get_default instance is None")
    return __default_bx_catalog


def bx_catalogs_has_default():
    global __default_bx_catalog
    return __default_bx_catalog is not None


# -----------------------------------------------------------------------------------------


class BxCatalogs(Things):
    """
    List of available bx_catalogs.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification):
        """"""

        bx_catalog_class = self.lookup_class(specification["type"])

        try:
            bx_catalog_object = bx_catalog_class(specification)
        except Exception as exception:
            raise RuntimeError(
                "unable to build bx_catalog object for type %s" % (bx_catalog_class)
            ) from exception

        return bx_catalog_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_catalogs.aiohttp":
            from dls_bxflow_lib.bx_catalogs.aiohttp import Aiohttp

            return Aiohttp

        elif class_type == "dls_bxflow_lib.bx_catalogs.ispyb":
            from dls_bxflow_lib.bx_catalogs.ispyb import Ispyb

            return Ispyb

        raise NotFound("unable to get bx_catalog class for type %s" % (class_type))
