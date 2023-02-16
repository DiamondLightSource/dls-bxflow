import logging

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Exceptions.
from dls_bxflow_api.exceptions import NotSet

# Base class for a Thing which has a name and traits.
from dls_bxflow_api.thing import Thing

# BxTasks for the bx_job.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks

logger = logging.getLogger(__name__)


class Base(Thing):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, thing_type, specification=None, predefined_uuid=None):
        Thing.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.__label = require(
            "%s specification" % (thing_type), specification, "label"
        )

        # Set directory if it happens to be in the specification.
        self.__directory = specification.get("directory")

        self.__data_label = None
        self.__workflow_uuid = None

        self.__bx_tasks = BxTasks()

    # -----------------------------------------------------------------------------
    def label(self):
        """Deprecated in favor of get_label."""
        return self.__label

    # -----------------------------------------------------------------------------
    def set_label(self, label):
        self.__label = label
        self.specification()["label"] = label

    def get_label(self):
        if self.__label is None:
            raise NotSet(f"{callsign(self)} label has not been set")
        return self.__label

    # -----------------------------------------------------------------------------
    def set_data_label(self, data_label):
        self.__data_label = data_label
        self.specification()["data_label"] = data_label

    def get_data_label(self):
        return self.__data_label

    # -----------------------------------------------------------------------------
    def set_directory(self, directory):
        self.__directory = directory

    def get_directory(self):
        if self.__directory is None:
            raise NotSet(f"{callsign(self)} directory has not been set")
        return self.__directory

    # -----------------------------------------------------------------------------
    def set_workflow_uuid(self, workflow_uuid):
        self.__workflow_uuid = workflow_uuid

    def get_workflow_uuid(self):
        return self.__workflow_uuid

    # -----------------------------------------------------------------------------
    def _get_bx_tasks(self):
        return self.__bx_tasks

    def _set_bx_tasks(self, bx_tasks):
        self.__bx_tasks = bx_tasks

    def _del_bx_tasks(self):
        del self.__bx_tasks

    bx_tasks = property(
        fget=_get_bx_tasks,
        fset=_set_bx_tasks,
        fdel=_del_bx_tasks,
        doc="The bx_tasks property.",
    )
