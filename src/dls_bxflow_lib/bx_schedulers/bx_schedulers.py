# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_bx_scheduler = None


def bx_schedulers_set_default(bx_scheduler):
    global __default_bx_scheduler
    __default_bx_scheduler = bx_scheduler


def bx_schedulers_get_default():
    global __default_bx_scheduler
    if __default_bx_scheduler is None:
        raise RuntimeError("bx_schedulers_get_default instance is None")
    return __default_bx_scheduler


# -----------------------------------------------------------------------------------------


class BxSchedulers(Things):
    """
    List of available bx_schedulers.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification):
        """"""

        bx_scheduler_class = self.lookup_class(specification["type"])

        try:
            bx_scheduler_object = bx_scheduler_class(specification)
        except Exception as exception:
            raise RuntimeError(
                "unable to build bx_scheduler object for type %s" % (bx_scheduler_class)
            ) from exception

        return bx_scheduler_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_schedulers.aiohttp":
            from dls_bxflow_lib.bx_schedulers.aiohttp import Aiohttp

            return Aiohttp

        if class_type == "dls_bxflow_lib.bx_schedulers.dask":
            from dls_bxflow_lib.bx_schedulers.dask import Dask

            return Dask

        if class_type == "dls_bxflow_lib.bx_schedulers.naive":
            from dls_bxflow_lib.bx_schedulers.naive import Naive

            return Naive

        if class_type == "dls_bxflow_lib.bx_schedulers.zocalo":
            from dls_bxflow_lib.bx_schedulers.zocalo import Zocalo

            return Zocalo

        raise NotFound("unable to get bx_scheduler class for type %s" % (class_type))
