import glob
import logging
import os
from typing import List

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.isodatetime import isodatetime_filename
from dls_utilpack.sanitize import sanitize

# Exceptions.
from dls_bxflow_api.exceptions import NotSet

# Base class for generic things.
from dls_bxflow_api.thing import Thing

logger = logging.getLogger(__name__)


class ExecutionOutput:
    """
    Object describing a file.
    """

    # Full name of the file.
    filename = None
    # Directory name.
    dirname = None
    # Base name (filename with directory removed).
    basename = None
    # Size in bytes.
    bytes = None

    def __init__(
        self,
        filename=None,
        dirname=None,
        basename=None,
        bytes=None,
    ):
        self.filename = filename
        self.dirname = dirname
        self.basename = basename
        self.bytes = bytes


class Base(Thing):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, thing_type, specification=None):
        Thing.__init__(self, thing_type, specification)

        self.__beamline = None
        self.__year = None
        self.__visit = None

    # ----------------------------------------------------------------------------------------
    def set_beamline(self, beamline):
        self.__beamline = beamline

    # ----------------------------------------------------------------------------------------
    def get_beamline(self):
        if self.__beamline is None:
            raise NotSet(f"{callsign(self)} has not been assigned beamline")
        return self.__beamline

    # ----------------------------------------------------------------------------------------
    def set_year(self, year):
        self.__year = year

    # ----------------------------------------------------------------------------------------
    def get_year(self):
        if self.__year is None:
            raise NotSet(f"{callsign(self)} has not been assigned year")
        return self.__year

    # ----------------------------------------------------------------------------------------
    def set_visit(self, visit):
        self.__visit = visit

    # ----------------------------------------------------------------------------------------
    def get_visit(self):
        if self.__visit is None:
            raise NotSet(f"{callsign(self)} has not been assigned visit")
        return self.__visit

    # ----------------------------------------------------------------------------------------
    def flatten_label_uuid(self, label, uuid):
        """"""

        label_part = sanitize(label[:64])

        random_part = uuid.split("-")[0]

        return f"{label_part}-{random_part}"

    # ----------------------------------------------------------------------------------------
    def flatten(self, capsule):
        """"""

        flattened = self.flatten_label_uuid(capsule.label(), capsule.uuid())

        return flattened

    # ----------------------------------------------------------------------------------------
    def pin_job_directory(self, bx_job):
        """
        Get a directory that is meant to be unique in the lab.
        Don't change a job's pinned directory on a second pin call.
        Using date-time-microsends isn't 100% guaranteed.
        TODO: Die if inserting a new job when its directory already exists.
        """

        try:
            directory = bx_job.get_directory()
            logger.debug(f"{callsign(self)} not overpinning job directory {directory}")
        except NotSet:
            directory = self.overpin_job_directory(bx_job)

        return directory

    # ----------------------------------------------------------------------------------------
    def overpin_job_directory(self, bx_job):
        """
        Get a directory that is meant to be unique in the lab.
        Using date-time-microsends isn't 100% guaranteed.
        Overpin is used when we want to change the job's directory after the initial pin.
        This is the case in a generic workflow which gets its name after it has created the job object.
        TODO: Die if inserting a new job when its directory already exists.
        """

        try:
            directory = bx_job.get_directory()
            # Pick off the date part from the previous job directory.
            datepart = directory.split(".")[-1]
        except NotSet:
            datepart = isodatetime_filename()

        directory = "%s/%s" % (
            self.get_workflows_anchor(),
            f"{sanitize(bx_job.label())}.{datepart}",
        )
        bx_job.set_directory(directory)
        logger.debug(f"{callsign(self)} pinned job directory {directory}")

        return directory

    # ----------------------------------------------------------------------------------------
    def pin_task_directory(self, bx_job, bx_task):
        """
        Tasks are assumed to have unique labels.
        Don't change a task's pinned directory on a second pin call.
        """

        try:
            directory = bx_task.get_directory()
            logger.debug(f"{callsign(self)} not overpinning task directory {directory}")
        except NotSet:
            directory = "%s/%s" % (
                bx_job.get_directory(),
                sanitize(bx_task.label()),
            )
            bx_task.set_directory(directory)
            logger.debug(f"{callsign(self)} pinned task directory {directory}")

        return directory

    # ----------------------------------------------------------------------------------------
    def get_runtime_execution_outputs(
        self, bx_task_directory: str
    ) -> List[ExecutionOutput]:
        """
        Find all the files in the tasks's directory.

        Args:
            bx_task_directory (str): directory assigned to the task

        Returns:
            List[ExecutionOutput]: list of objects describing the files
        """

        filenames = glob.glob(f"{bx_task_directory}/*")
        execution_outputs = []

        for filename in filenames:
            # Make a convenient object with attributes about the file.
            execution_output = ExecutionOutput(
                filename=filename,
                dirname=os.path.dirname(filename),
                basename=os.path.basename(filename),
                bytes=os.path.getsize(filename),
            )
            execution_outputs.append(execution_output)

        return execution_outputs
