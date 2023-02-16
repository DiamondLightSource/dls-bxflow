import logging

from dls_bxflow_api.bx_datafaces.context import Context as ApiBxDatafaceContext

# Base class for an asyncio context
from dls_bxflow_lib.bx_contexts.base import Base as ContextBase

# Things created in the context.
from dls_bxflow_lib.bx_datafaces.bx_datafaces import BxDatafaces

logger = logging.getLogger(__name__)


thing_type = "dls_bxflow_lib.bx_datafaces.context"


class Context(ContextBase):
    """
    Asyncio context for a bx_dataface server object.
    On entering, it creates the object according to the specification (a dict).
    If configured, it starts the server as a coroutine, thread or process.
    On exiting, it commands the server to shut down.

    The enter and exit methods are exposed for use during testing.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification):
        ContextBase.__init__(self, thing_type, specification)

        self.__api_bx_dataface_context = None

    # ----------------------------------------------------------------------------------------
    async def aenter(self):
        """ """

        # Build the object according to the specification.
        self.server = BxDatafaces().build_object(self.specification())

        if self.context_specification.get("start_as") == "coro":
            await self.server.activate_coro()

        elif self.context_specification.get("start_as") == "thread":
            await self.server.start_thread()

        elif self.context_specification.get("start_as") == "process":
            await self.server.start_process()

        self.__api_bx_dataface_context = ApiBxDatafaceContext(self.specification())
        await self.__api_bx_dataface_context.aenter()

    # ----------------------------------------------------------------------------------------
    async def aexit(self):
        """ """

        if self.server is not None:
            # Put in request to shutdown the server.
            await self.server.client_shutdown()

        if self.__api_bx_dataface_context is not None:
            await self.__api_bx_dataface_context.aexit()
