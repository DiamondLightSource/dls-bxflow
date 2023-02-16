import logging

# Utilities.
from dls_utilpack.require import require

# Object managing bx_datafaces.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

logger = logging.getLogger(__name__)


class DatafaceClient:
    def __init__(self, configuration):
        self.__configuration = configuration

    # ----------------------------------------------------------------------------------------
    async def __aenter__(self):
        """ """

        # ------------------------------------------------------------------------
        bx_dataface_specification = require(
            "configuration", self.__configuration, "bx_dataface_specification"
        )

        # Make a bx_dataface client.
        from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_set_default
        from dls_bxflow_lib.bx_datafaces.bx_datafaces import BxDatafaces

        bx_datafaces = BxDatafaces()
        bx_dataface = bx_datafaces.build_object(bx_dataface_specification)
        bx_datafaces_set_default(bx_dataface)

    # ----------------------------------------------------------------------------------------
    async def __aexit__(self, type, value, traceback):
        """ """
        # Put in request to shutdown the dataface.
        await bx_datafaces_get_default().close_client_session()
