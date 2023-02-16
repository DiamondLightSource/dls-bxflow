# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_gaml = None


def bx_gamls_set_default(gaml):
    global __default_gaml
    __default_gaml = gaml


def bx_gamls_get_default():
    global __default_gaml
    if __default_gaml is None:
        raise RuntimeError("bx_gamls_get_default instance is None")
    return __default_gaml


# -----------------------------------------------------------------------------------------


class BxGamls(Things):
    """
    List of available bx_gamls.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification):
        """"""

        gaml_class = self.lookup_class(specification["type"])

        try:
            gaml_object = gaml_class(specification)
        except Exception as exception:
            raise RuntimeError(
                "unable to build gaml object for type %s" % (gaml_class)
            ) from exception

        return gaml_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_gamls.html":
            from dls_bxflow_lib.bx_gamls.html import Html

            return Html

        raise NotFound("unable to get gaml class for type %s" % (class_type))
