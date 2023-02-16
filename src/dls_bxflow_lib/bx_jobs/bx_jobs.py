# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

# Parameters.
from dls_bxflow_run.bx_variables.bx_variables import BxVariables

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_job = None


def bx_jobs_set_default(job):
    global __default_job
    __default_job = job


def bx_jobs_get_default():
    global __default_job
    if __default_job is None:
        raise RuntimeError("bx_jobs_get_default instance is None")
    return __default_job


# -----------------------------------------------------------------------------------------


class BxJobs(Things):
    """
    List of available bx_jobs.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name="bx_jobs"):
        Things.__init__(self, name)
        self.__variables = BxVariables()

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification=None, predefined_uuid=None):
        """"""

        if specification is None:
            specification = {"type": "dls_bxflow_lib.bx_jobs.standard"}

        # If a string, parse for json, yaml or whatever.
        specification = self.parse_specification(specification)

        bx_job_class = self.lookup_class(specification["type"])

        try:
            bx_job_object = bx_job_class(specification, predefined_uuid=predefined_uuid)
        except Exception as exception:
            raise RuntimeError(
                "unable to build bx_job object for type %s" % (bx_job_class)
            ) from exception

        return bx_job_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_jobs.standard":
            from dls_bxflow_lib.bx_jobs.standard import Standard

            return Standard

        raise NotFound("unable to get bx_job class for type %s" % (class_type))
