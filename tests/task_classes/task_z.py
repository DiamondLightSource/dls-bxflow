import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TaskZ:
    """
    This is a user-defined class which gets instantiated and called at runtime.
    """

    async def run(self):
        logger.info("running TaskZ")
        # logger.info(describe("os.environ", dict(os.environ)))
