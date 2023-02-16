# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_composer = None


def bx_composers_set_default(composer):
    global __default_composer
    __default_composer = composer


def bx_composers_get_default():
    global __default_composer
    if __default_composer is None:
        raise RuntimeError("bx_composers_get_default instance is None")
    return __default_composer


def bx_composers_has_default():
    global __default_composer
    return __default_composer is not None


# -----------------------------------------------------------------------------------------


class BxComposers(Things):
    """
    List of available bx_composers.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification):
        """"""

        composer_class = self.lookup_class(specification["type"])

        try:
            composer_object = composer_class(specification)
        except Exception as exception:
            raise RuntimeError(
                "unable to build composer object for type %s" % (composer_class)
            ) from exception

        return composer_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_composers.html":
            from dls_bxflow_lib.bx_composers.html import Html

            return Html

        elif class_type == "dls_bxflow_lib.bx_composers.text":
            from dls_bxflow_lib.bx_composers.text import Text

            return Text

        raise NotFound("unable to get composer class for type %s" % (class_type))
