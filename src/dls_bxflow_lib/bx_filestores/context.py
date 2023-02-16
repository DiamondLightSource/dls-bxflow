import logging

# Base class for an asyncio context
from dls_bxflow_lib.bx_contexts.base import Base as ContextBase

# Things created in the context.
from dls_bxflow_lib.bx_filestores.bx_filestores import (
    BxFilestores,
    bx_filestores_set_default,
)

logger = logging.getLogger(__name__)


thing_type = "dls_bxflow_lib.bx_filestores.context"


class Context(ContextBase):
    """
    Asyncio context for a bx_filestore object.
    On entering, it creates the object according to the specification (a dict).
    If configured, it starts the server as a coroutine, thread or process.
    On exiting, it commands the server to shut down and closes client connection.

    The enter and exit methods are exposed for use during testing.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification):
        ContextBase.__init__(self, thing_type, specification)

    # ----------------------------------------------------------------------------------------
    async def aenter(self):
        """ """

        # Build the object according to the specification.
        self.server = BxFilestores().build_object(self.specification())

        # If there is more than one filestore, the last one defined will be the default.
        bx_filestores_set_default(self.server)

    # ----------------------------------------------------------------------------------------
    async def aexit(self):
        """ """

        # Clear the global variable.  Important between pytests.
        bx_filestores_set_default(None)
