import logging

# Base class for an asyncio context
from dls_bxflow_lib.bx_contexts.base import Base as ContextBase

# Things created in the context.
from dls_bxflow_lib.bx_schedulers.bx_schedulers import (
    BxSchedulers,
    bx_schedulers_set_default,
)

logger = logging.getLogger(__name__)


thing_type = "dls_bxflow_lib.bx_schedulers.context"


class Context(ContextBase):
    """
    Asyncio context for a bx_scheduler object.
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
        self.server = BxSchedulers().build_object(self.specification())

        # If there is more than one scheduler, the last one defined will be the default.
        bx_schedulers_set_default(self.server)

        if self.context_specification.get("start_as") == "coro":
            await self.server.activate_coro()

        elif self.context_specification.get("start_as") == "thread":
            await self.server.start_thread()

        elif self.context_specification.get("start_as") == "process":
            await self.server.start_process()

    # ----------------------------------------------------------------------------------------
    async def aexit(self):
        """ """

        if self.server is not None:
            start_as = self.context_specification.get("start_as")
            if start_as in ["coro", "thread", "process"]:
                # Put in request to shutdown the server.
                await self.server.client_shutdown()
            else:
                await self.server.close_client_session()

        # Clear the global variable.  Important between pytests.
        bx_schedulers_set_default(None)
