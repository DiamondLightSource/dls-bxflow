# Use standard logging in this module.
import logging
import os

# Base class for femtocheck subcommands.
from dls_bxflow_cli.subcommands.base import Base

logger = logging.getLogger()


# --------------------------------------------------------------
class FindXanes(Base):
    """
    Find Xanes files produced by Luigi.
    """

    def __init__(self, args):
        super().__init__(args)

    # ----------------------------------------------------------
    def run(self):
        """"""

        top = "/dls/i14/data/2022"
        top_level = len(top.split("/"))
        for root, directories, files in os.walk(top):
            if root.endswith("auto"):
                for name in directories:
                    logger.info(f"{root}/{name}")
                # for name in files:
                #     logger.info(f"{root}/{name}")

            root_level = len(root.split("/"))

            if root_level == top_level + 1:
                has_processing = "processing" in directories
                directories.clear()
                if has_processing:
                    directories.append("processing")

            if root_level == top_level + 2:
                has_auto = "auto" in directories
                directories.clear()
                if has_auto:
                    directories.append("auto")

            if root_level == top_level + 3:
                directories.clear()

    # ----------------------------------------------------------
    def add_arguments(parser):

        return parser
