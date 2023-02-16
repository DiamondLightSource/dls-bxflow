# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_bx_gui = None


def bx_guis_set_default(bx_gui):
    global __default_bx_gui
    __default_bx_gui = bx_gui


def bx_guis_get_default():
    global __default_bx_gui
    if __default_bx_gui is None:
        raise RuntimeError("bx_guis_get_default instance is None")
    return __default_bx_gui


def bx_guis_has_default():
    global __default_bx_gui
    return __default_bx_gui is not None


# -----------------------------------------------------------------------------------------


class BxGuis(Things):
    """
    List of available bx_guis.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification):
        """"""

        bx_gui_class = self.lookup_class(specification["type"])

        try:
            bx_gui_object = bx_gui_class(specification)
        except Exception as exception:
            raise RuntimeError(
                "unable to build bx_gui object for type %s" % (bx_gui_class)
            ) from exception

        return bx_gui_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_guis.aiohttp":
            from dls_bxflow_lib.bx_guis.aiohttp import Aiohttp

            return Aiohttp

        elif class_type == "dls_bxflow_lib.bx_guis.curses":
            from dls_bxflow_lib.bx_guis.curses import Curses

            return Curses

        raise NotFound("unable to get bx_gui class for type %s" % (class_type))
