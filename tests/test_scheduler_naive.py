import logging
import os

# Object managing datafaces.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Exceptions.
from dls_bxflow_api.exceptions import TransientError

# Remex things.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing launchers.
from dls_bxflow_lib.bx_launchers.states import States as BxLauncherStates

# BxSchedulers manager.
from dls_bxflow_lib.bx_schedulers.bx_schedulers import BxSchedulers

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
class TestSchedulerNaive:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"

        # Good test.
        SchedulerNaiveTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class SchedulerNaiveTester(BaseContextTester):
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
        # bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_gui_specification")
        bx_configurator.remove("dls_servbase_dataface_specification")

        # Load the configuration file and resolve the substitutions.
        context_configuration = await bx_configurator.load()
        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:
            # Make a scheduler object.
            naive_scheduler_configuration = bx_configurator.require(
                "bx_scheduler_specification"
                ".type_specific_tbd"
                ".actual_bx_scheduler_specification"
            )
            naive_scheduler = BxSchedulers().build_object(naive_scheduler_configuration)

            # Make a task specification.
            task_specification = {}

            # ----------------------------------------------------------------------
            case = "single hint, found"
            task_specification[RemexKeywords.HINTS] = {
                RemexKeywords.CLUSTER: RemexClusters.LOCAL
            }
            launcher_record = await naive_scheduler.select_launcher_record(
                bx_task_uuid, task_specification
            )
            assert launcher_record["uuid"] == "popener-01", case

            # ----------------------------------------------------------------------
            case = "mulitple hints, found second"
            task_specification[RemexKeywords.HINTS] = {
                RemexKeywords.CLUSTER: [RemexClusters.HAMILTON, RemexClusters.LOCAL]
            }
            launcher_record = await naive_scheduler.select_launcher_record(
                bx_task_uuid, task_specification
            )
            assert launcher_record is not None
            assert launcher_record["uuid"] == "popener-01", case

            # ----------------------------------------------------------------------
            case = "multiple hints, found None"
            task_specification[RemexKeywords.HINTS] = {
                RemexKeywords.CLUSTER: [RemexClusters.HAMILTON, RemexClusters.SCIENCE]
            }
            try:
                launcher_record = await naive_scheduler.select_launcher_record(
                    bx_task_uuid, task_specification
                )
                assert False, case
            except TransientError as exception:
                assert "[NOLAUNCH2]" in str(exception)

            # ----------------------------------------------------------------------
            case = "single hint, found a luncher but it is busy"
            row = {"uuid": "popener-01", "state": BxLauncherStates.BUSY}
            await bx_datafaces_get_default().update_bx_launcher(row)
            task_specification[RemexKeywords.HINTS] = {
                RemexKeywords.CLUSTER: RemexClusters.LOCAL
            }
            try:
                launcher_record = await naive_scheduler.select_launcher_record(
                    bx_task_uuid, task_specification
                )
                assert False, case
            except TransientError as exception:
                assert "[NOLAUNCH2]" in str(exception)

            # ----------------------------------------------------------------------
            case = "single hint, found no idle launchers"
            row = {"uuid": "popener-01", "state": BxLauncherStates.BUSY}
            row = {"uuid": "qsubber-01", "state": BxLauncherStates.BUSY}
            await bx_datafaces_get_default().update_bx_launcher(row)
            task_specification[RemexKeywords.HINTS] = {
                RemexKeywords.CLUSTER: RemexClusters.LOCAL
            }
            try:
                launcher_record = await naive_scheduler.select_launcher_record(
                    bx_task_uuid, task_specification
                )
                assert False, case
            except TransientError as exception:
                assert "[NOLAUNCH1]" in str(exception)
