import asyncio
import copy
import json
import logging
import os
import time

# Utilities.
from dls_utilpack.qualname import qualname

# Object managing datafaces.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Exceptions.
from dls_bxflow_api.exceptions import CapacityReached

# Remote execution.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing launchers.
from dls_bxflow_lib.bx_filestores.bx_filestores import bx_filestores_get_default

# BxJobs manager.
from dls_bxflow_lib.bx_jobs.bx_jobs import bx_jobs_get_default

# Object managing launchers.
from dls_bxflow_lib.bx_launchers.bx_launchers import BxLaunchers

# BxTasks manager.
from dls_bxflow_run.bx_tasks.states import States as BxTaskStates

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


bx_job_uuid = "00001-2222-3333-4444-5555555"

bx_job_capsule = (
    "my good job",
    bx_job_uuid,
)

bx_task_uuid = "%05d-2222-3333-4444-5555555"
bx_task_label = "my launcher task %d"

dummy_task_delay = 3.0
nap = 1.0


# ----------------------------------------------------------------------------------------
class TestLauncherCapacity:
    def test(self, constants, logging_setup, output_directory):
        """ """

        # Configuration file to use.
        configuration_file = "tests/configurations/backend.yaml"

        # Good test.
        expected_exit_code = 0
        LauncherCapacityTester(
            expected_exit_code,
            None,
        ).main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class LauncherCapacityTester(BaseContextTester):
    """
    Class to test the launcher.
    """

    def __init__(
        self,
        expected_exit_code,
        expected_error_lines,
    ):
        BaseContextTester.__init__(self)

        self.__expected_exit_code = expected_exit_code
        self.__expected_error_lines = expected_error_lines

    async def _main_coroutine(self, constants, output_directory):
        """ """
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

        task_count_max = 3

        # Keep only the one desired bx_launcher_specification.
        self.__desired_bx_launcher_specification = "bx_launcher_popener_specification"
        bx_launcher_specifications = context_configuration["bx_launcher_specifications"]
        bx_launcher_specifications.clear()
        bx_launcher_specifications.append(self.__desired_bx_launcher_specification)

        # Modify the configured maximum task count.
        bx_configurator.require(
            "bx_launcher_popener_specification"
            ".type_specific_tbd"
            ".actual_bx_launcher_specification"
            ".type_specific_tbd"
        )["task_count_max"] = task_count_max

        bx_context = BxContexts().build_object(context_configuration)

        # Directory where the task should write its files.
        bx_task_directory = f"{output_directory}/task_%d"
        # Files discovered after the task has run.
        execution_outputs = []

        task_count_in_job = task_count_max + 2
        bx_task_specifications = []
        bx_task_execution_outputs = []
        async with bx_context:
            launcher = BxLaunchers().build_object(
                bx_configurator.require(self.__desired_bx_launcher_specification)
            )

            try:
                bx_job_specification = copy.deepcopy(
                    bx_jobs_get_default().specification()
                )
                bx_job_specification["label"] = "my/good/job_%d"
                bx_job_specification["directory"] = "my_good_job"

                for i in range(task_count_in_job):
                    # This file will be written by dummy outside the filestore.
                    dummy_outfile = f"{output_directory}/dummy_datafile_{i}.json"
                    bx_task_specification = {
                        "uuid": bx_task_uuid % (i),
                        "type": "dls_bxflow_run.bx_tasks.dummy",
                        "label": bx_task_label % (i),
                        "directory": bx_task_directory % (i),
                        "bx_job_uuid": bx_job_uuid,
                        RemexKeywords.HINTS: {
                            RemexKeywords.CLUSTER: RemexClusters.LOCAL,
                            "ptypy_mpi": {"num_gpu": 8},
                        },
                        "type_specific_tbd": {
                            "outfile": dummy_outfile,
                            "delay": dummy_task_delay,
                        },
                    }

                    bx_task_specifications.append(bx_task_specification)

                # Define the tasks in the database.
                await bx_datafaces_get_default().set_bx_tasks(bx_task_specifications)

                for i in range(task_count_in_job):
                    time0 = time.time()
                    # We will need to wait for one task delay, plus extra per task submitted.
                    timeout = dummy_task_delay + 1.0 * (
                        task_count_in_job - task_count_max
                    )
                    while True:
                        try:
                            # Submit the bx_task.
                            await launcher.submit(
                                bx_job_uuid,
                                bx_job_specification,
                                bx_task_uuid % (i),
                                bx_task_specifications[i],
                            )
                            break
                        except Exception as exception:
                            if qualname(CapacityReached) in str(exception):
                                logger.info(
                                    f"general {exception}, will nap and try again"
                                )
                            else:
                                raise

                        await asyncio.sleep(nap)

                        if time.time() - time0 > timeout:
                            raise RuntimeError(
                                f"bx_tasks submission not finished within {timeout} seconds"
                            )

                time0 = time.time()
                timeout = 10.0
                while True:
                    # Let the server harvest the running task.
                    await launcher.harvest()

                    # Check the task status.
                    task_records = await bx_datafaces_get_default().get_bx_tasks(
                        bx_job_uuid
                    )

                    unfinished_count = 0
                    states = []
                    for task_record in task_records:
                        # Task finished?
                        if task_record["state"] != BxTaskStates.FINISHED:
                            unfinished_count += 1
                        states.append(str(task_record["state"]))

                    logger.info("states are [%s]" % (", ".join(states)))

                    if unfinished_count == 0:
                        break

                    await asyncio.sleep(nap)

                    if time.time() - time0 > timeout:
                        raise RuntimeError(
                            f"bx_tasks completion not finished within {timeout} seconds"
                        )

                    # Grab outputs from all the tasks while still in the bx_context.
                    for i in range(task_count_in_job):
                        execution_outputs = (
                            bx_filestores_get_default().get_runtime_execution_outputs(
                                bx_task_directory % (i)
                            )
                        )
                        bx_task_execution_outputs.append(execution_outputs)

            finally:
                await launcher.close_client_session()

            for i in range(task_count_in_job):
                execution_outputs = bx_task_execution_outputs[i]

                assert len(task_record["error_lines"]) == 0

                # Verify all the dummy output file contents.
                dummy_outfile = f"{output_directory}/dummy_datafile_{i}.json"
                with open(dummy_outfile, "r") as stream:
                    dummy_output = json.load(stream)
                    assert dummy_output["bx_task_label"] == bx_task_label % (i)

                # Verify the existence of the residual files.
                self.assert_execution_residuals(execution_outputs)
