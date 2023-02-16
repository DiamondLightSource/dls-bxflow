import logging

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class GdaParserBase:
    """
    Base class for GDA message parsing.
    """

    def is_scan_finish(self, message, headers):
        # TODO: Figure out some good things to do in the gda parser base class.
        pass

        # # Do some basic checks on the message,
        # # TODO: Use symbolic constants for gda scan topic message fields.
        # if "status" not in message:
        #     logger.debug("ignoring stomp message because no status field")
        #     return False

        # status = message["status"]
        # if status != "FINISHED":
        #     logger.debug(f"ignoring stomp message with status {status}")
        #     return False

        # if "filePath" not in message:
        #     logger.debug("ignoring stomp message: no filePath field")
        #     return False
