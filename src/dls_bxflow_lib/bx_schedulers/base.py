import logging

# Base class which maps flask bx_tasks to methods.
from dls_utilpack.thing import Thing

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------
class Base(Thing):
    """
    Object representing a bx_scheduler.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, thing_type, specification=None, predefined_uuid=None):
        Thing.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)
