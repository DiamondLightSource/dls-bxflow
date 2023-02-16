import json
import logging

# Workflow base class.
from dls_bxflow_lib.bx_collectors.gda_parser_base import GdaParserBase

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class GdaParser(GdaParserBase):
    """
    This workflow has one task.
    """

    # ------------------------------------------------------------------------------------
    def parse(self, message, headers):
        """
        Parse the gda message.
        """

        logger.info(
            "gda_parser returning message\n%s" % (json.dumps(message, indent=4))
        )
        # The message passed during testing has all what we need to return.
        return message
