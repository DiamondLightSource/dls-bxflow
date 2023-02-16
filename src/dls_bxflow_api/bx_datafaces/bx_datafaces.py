# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_bx_dataface = None


def bx_datafaces_set_default(bx_dataface):
    global __default_bx_dataface
    __default_bx_dataface = bx_dataface


def bx_datafaces_get_default():
    global __default_bx_dataface
    if __default_bx_dataface is None:
        raise RuntimeError("bx_datafaces_get_default instance is None")
    return __default_bx_dataface


# -----------------------------------------------------------------------------------------


class BxDatafaces(Things):
    """
    List of available bx_datafaces.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification):
        """"""

        bx_dataface_class = self.lookup_class(specification["type"])

        try:
            bx_dataface_object = bx_dataface_class(specification)
        except Exception as exception:
            raise RuntimeError(
                "unable to build bx_dataface object for type %s" % (bx_dataface_class)
            ) from exception

        return bx_dataface_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_datafaces.aiohttp":
            from dls_bxflow_api.bx_datafaces.aiohttp import Aiohttp

            return Aiohttp

        raise NotFound("unable to get bx_dataface class for type %s" % (class_type))
