import logging
import os

# Utilities.
from dls_utilpack.describe import describe

# Object managing datafaces.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing launchers.
from dls_bxflow_lib.bx_launchers.states import States as BxLauncherStates

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


bx_job_uuid = "00001-2222-3333-4444-5555555"

bx_job_capsule = (
    "my good job",
    bx_job_uuid,
)

bx_task_uuid = "00002-2222-3333-4444-5555555"
bx_task_label = "my launcher task (part 1)"


# ----------------------------------------------------------------------------------------
class TestLauncherMultiple:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"

        # Good test.
        LauncherMultipleTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class LauncherMultipleTester(BaseContextTester):
    """
    Class to test the launcher running jobs directly, without a workflow involved.
    """

    def __init__(
        self,
    ):
        BaseContextTester.__init__(self)

    async def _main_coroutine(self, constants, output_directory):
        """ """

        # Make the fake qsub findable in the path.
        os.environ["PATH"] = "%s:%s" % (os.path.dirname(__file__), os.environ["PATH"])

        # Supply environment variable for substitution in the bx configurator.
        os.environ["OUTPUT_DIRECTORY"] = output_directory

        bx_configurator = self.get_bx_configurator()

        # Don't start the services we don't need for this test.
        bx_configurator.remove("dls_servbase_dataface_specification")
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_gui_specification")

        # Load the configuration file and resolve the substitutions.
        context_configuration = await bx_configurator.load()
        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:
            launcher_records = await bx_datafaces_get_default().get_bx_launchers(
                [BxLauncherStates.IDLE]
            )

            if len(launcher_records) == 0:
                logger.warning("launchers all busy")
                return

            logger.debug(describe("launcher records", launcher_records))
