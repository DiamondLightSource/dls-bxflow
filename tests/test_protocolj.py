import logging

import pytest

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing news.
from dls_bxflow_lib.bx_news.bx_news import bx_news_get_default
from dls_bxflow_lib.bx_news.constants import Topics

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestProtocoljErrorsLaptop:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        ProtocoljErrorsTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class ProtocoljErrorsTester(BaseContextTester):
    """
    Class to test the news.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

        # Let the testing base class make an appropriate configurator.
        bx_configurator = self.get_bx_configurator()

        # Make a mainiac instance and let it configure mpqueue logging.
        # mainiac = Mainiac("protocolj_errors_tester")
        # mainiac.configure_logging({"mpqueue": {"enabled": False}})
        # mainiac.configure_logging({"mpqueue": {"enabled": False}})

        # Tell the configurator about the mpqueue logging.
        # bx_configurator.set_logging_mpqueue(mainiac.mpqueue)

        # Don't build the higher level servers.
        bx_configurator.remove("bx_filestore_specification")
        bx_configurator.remove("bx_dataface_specification")
        bx_configurator.remove("bx_job_specification")
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_launcher_specifications")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()
        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:
            self.topic = Topics.BXJOB_WAS_ENABLED

            with pytest.raises(RuntimeError) as exception_info:
                await bx_news_get_default().test_uncaught("uncaught.1")

            assert "uncaught.1" in str(exception_info.value)
