import json
import logging

import pytest

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing logstore.
from dls_bxflow_lib.bx_logstores.bx_logstores import bx_logstores_get_default

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
@pytest.mark.skipif("not config.getoption('graylog')")
class TestLogstore:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        LogstoreTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class LogstoreTester(BaseContextTester):
    """
    Class to test the logstore.
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
        bx_configurator.remove("bx_news_specification")
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
            log_lines = await bx_logstores_get_default().query([])

            logger.info(
                "first response\n[[[[[[[[[[\n%s\n]]]]]]]]]]"
                % (json.dumps(log_lines, indent=4))
            )

            # ---------------------------------------------------------------------------
            # Check if date range works.
            log_line = log_lines[-2]

            where_ands = [
                {
                    "field": "timestamp",
                    "operator": ">",
                    "operand": log_line["timestamp"],
                }
            ]
            new_log_lines = await bx_logstores_get_default().query(where_ands)

            logger.info(f"got {len(new_log_lines)} date range lines")

            logger.info(
                "date range\n[[[[[[[[[[\n%s\n]]]]]]]]]]"
                % (json.dumps(new_log_lines, indent=4))
            )

            assert len(new_log_lines) == 2, "one new line using last date but one"

            # ---------------------------------------------------------------------------
            # Check if no more lines
            log_line = log_lines[-1]

            where_ands = [
                {
                    "field": "timestamp",
                    "operator": ">",
                    "operand": log_line["timestamp"],
                }
            ]
            new_log_lines = await bx_logstores_get_default().query(where_ands)

            logger.info(f"got {len(new_log_lines)} new lines")

            logger.info(
                "no new lines response\n[[[[[[[[[[\n%s\n]]]]]]]]]]"
                % (json.dumps(new_log_lines, indent=4))
            )

            assert len(new_log_lines) == 1, "no new lines using last date"
