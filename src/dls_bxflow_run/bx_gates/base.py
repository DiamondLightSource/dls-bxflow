import logging

# Utilities.
from dls_utilpack.require import require

# Base class for a Thing which has a name and traits.
from dls_bxflow_api.thing import Thing

# States of bx_gates.
from dls_bxflow_run.bx_gates.states import States

logger = logging.getLogger(__name__)


class Base(Thing):
    """
    Base class for a BxGate.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, thing_type, specification=None, predefined_uuid=None):
        Thing.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.__label = require(
            "%s specification" % (thing_type), specification, "label"
        )

        self.__bx_task_uuid = None

        self.state(States.CLOSED)

    # -----------------------------------------------------------------------------
    def label(self):
        return self.__label

    # -----------------------------------------------------------------------------
    # TODO: Make the setter/getter pattern use kwargs check to determine mode.

    def bx_task_uuid(self, bx_task_uuid=None):

        if bx_task_uuid is not None:
            self.__bx_task_uuid = bx_task_uuid

        return self.__bx_task_uuid
