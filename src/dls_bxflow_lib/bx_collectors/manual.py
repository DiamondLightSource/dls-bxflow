import logging

# Base class for bx_collector instances.
from dls_bxflow_lib.bx_collectors.base import Base as BxCollectorBase

# Utilities.

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_collectors.popener"


# ------------------------------------------------------------------------------------------
class Manual(BxCollectorBase):
    """
    Object representing a bx_collector which launches a task using popen for onboard execution.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification, predefined_uuid=None):
        BxCollectorBase.__init__(
            self, thing_type, specification, predefined_uuid=predefined_uuid
        )

    # ----------------------------------------------------------------------------------------
    async def activate(self):
        """"""
        pass

    # ----------------------------------------------------------------------------------------
    async def deactivate(self):
        """"""
        pass

    # ----------------------------------------------------------------------------------------
    async def fire(
        self,
        message,
    ):
        """"""

        await BxCollectorBase.trigger(
            self,
            message["workflow_filename_classname"],
            **message["workflow_constructor_kwargs"],
        )
