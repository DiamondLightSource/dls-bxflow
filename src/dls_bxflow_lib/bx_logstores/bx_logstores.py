# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_bx_logstore = None


def bx_logstores_set_default(bx_logstore):
    global __default_bx_logstore
    __default_bx_logstore = bx_logstore


def bx_logstores_get_default():
    global __default_bx_logstore
    if __default_bx_logstore is None:
        raise RuntimeError("bx_logstores_get_default instance is None")
    return __default_bx_logstore


def bx_logstores_has_default():
    global __default_bx_logstore
    return __default_bx_logstore is not None


class BxLogstores(Things):
    """
    List of available bx_logstores.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name="bx_logstores"):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification, predefined_uuid=None):
        """"""

        bx_logstore_class = self.lookup_class(specification["type"])

        try:
            bx_logstore_object = bx_logstore_class(
                specification, predefined_uuid=predefined_uuid
            )
        except Exception as exception:
            raise RuntimeError(
                "unable to build bx_logstore object of class %s"
                % (bx_logstore_class.__name__)
            ) from exception

        return bx_logstore_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_logstores.aiohttp":
            from dls_bxflow_lib.bx_logstores.aiohttp import Aiohttp

            return Aiohttp

        elif class_type == "dls_bxflow_lib.bx_logstores.graylogger":
            from dls_bxflow_lib.bx_logstores.graylogger import Graylogger

            return Graylogger

        raise NotFound("unable to get bx_logstore class for %s" % (class_type))
