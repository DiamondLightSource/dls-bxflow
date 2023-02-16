import logging

# Utilities.
from dls_utilpack.require import require

logger = logging.getLogger(__name__)


class Filestore:
    def __init__(self, configuration):
        self.__configuration = configuration

    # ----------------------------------------------------------------------------------------
    async def __aenter__(self):
        """ """

        # ------------------------------------------------------------------------
        bx_filestore_specification = require(
            "configuration", self.__configuration, "bx_filestore_specification"
        )

        # Make a bx_filestore.
        from dls_bxflow_lib.bx_filestores.bx_filestores import (
            BxFilestores,
            bx_filestores_set_default,
        )

        bx_filestores = BxFilestores()
        bx_filestore = bx_filestores.build_object(bx_filestore_specification)
        bx_filestores_set_default(bx_filestore)

    # ----------------------------------------------------------------------------------------
    async def __aexit__(self, type, value, traceback):
        """ """
        pass
