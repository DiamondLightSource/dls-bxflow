import logging

# Base class which maps flask requests to methods.
from dls_bxflow_lib.bx_jobs.base import Base

# Things created in the context.
from dls_bxflow_lib.bx_jobs.bx_jobs import BxJobs, bx_jobs_set_default

logger = logging.getLogger(__name__)


thing_type = "dls_bxflow_lib.bx_jobs.context"


class Context(Base):
    """
    Object representing an event bx_dataface connection.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification):
        Base.__init__(self, thing_type, specification)

    # ----------------------------------------------------------------------------------------
    async def __aenter__(self):
        """ """

        await self.aenter()

    # ----------------------------------------------------------------------------------------
    async def __aexit__(self, type, value, traceback):
        """ """

        await self.aexit()

    # ----------------------------------------------------------------------------------------
    async def aenter(self):
        """ """

        thing = BxJobs().build_object(self.specification())

        bx_jobs_set_default(thing)

    # ----------------------------------------------------------------------------------------
    async def aexit(self):
        """ """

        pass
