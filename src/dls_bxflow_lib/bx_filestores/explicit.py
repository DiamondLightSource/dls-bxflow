import logging

# Utilities.
from dls_utilpack.callsign import callsign

# Exceptions.
from dls_bxflow_api.exceptions import NotSet

# Base class for filestore things.
from dls_bxflow_lib.bx_filestores.base import Base as BxFilestoreBase

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_filestores.explicit"


class ExecutionOutput:
    bx_job_capsule = None
    bx_task_capsule = None
    filename = None
    dirname = None
    basename = None
    bytes = None

    def __init__(
        self,
        bx_job_capsule=None,
        bx_task_capsule=None,
        filename=None,
        dirname=None,
        basename=None,
        bytes=None,
    ):
        self.bx_job_capsule = bx_job_capsule
        self.bx_task_capsule = bx_task_capsule
        self.filename = filename
        self.dirname = dirname
        self.basename = basename
        self.bytes = bytes


class Explicit(BxFilestoreBase):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        BxFilestoreBase.__init__(self, thing_type, specification)

        type_specific_tbd = self.specification().get("type_specific_tbd", {})

        self.__directory = None

        # Save the things which might be in the specification.
        self.set_directory(type_specific_tbd.get("directory"))
        self.set_beamline(type_specific_tbd.get("beamline"))
        self.set_year(type_specific_tbd.get("year"))
        self.set_visit(type_specific_tbd.get("visit"))

    # ----------------------------------------------------------------------------------------
    def set_directory(self, directory):
        logger.debug(
            f"{callsign(self)} overwriting {self.__directory} with {directory}"
        )
        self.__directory = directory

    # ----------------------------------------------------------------------------------------
    def get_directory(self):
        if self.__directory is None:
            raise NotSet(f"{callsign(self)} has not been assigned a directory")
        return self.__directory

    # ----------------------------------------------------------------------------------------
    def get_workflows_anchor(self):
        """ """

        return self.get_directory()
