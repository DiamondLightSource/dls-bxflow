import logging

# Object managing datafaces.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Results composers.
from dls_bxflow_lib.bx_composers.bx_composers import BxComposers

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class TestComposer:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/backend.yaml"
        ComposerTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class ComposerTester(BaseContextTester):
    """
    Class to test the composer.
    """

    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()

        # Don't build a launcher or scheduler or catalog.
        bx_configurator.remove("bx_launcher_specifications")
        bx_configurator.remove("bx_scheduler_specification")
        bx_configurator.remove("bx_catalog_specification")
        bx_configurator.remove("bx_collector_specification")
        bx_configurator.remove("bx_gui_specification")

        context_configuration = await bx_configurator.load()
        bx_context = BxContexts().build_object(context_configuration)

        async with bx_context:

            # -----------------------------------------------------------------
            # Add some jobs.
            bx_job_dicts = []
            bx_workflow_dicts = []
            data_count = 4
            for data_number in range(data_count):
                job_count = 4
                for job_number in range(job_count):
                    if data_number == 1 and job_number == 1:
                        pass
                    else:
                        bx_job_uuid = f"job_{data_number}x{job_number}_1"
                        bx_workflow_uuid = f"workflow_{data_number}x{job_number}_1"
                        bx_job_dict = {
                            "created_on": "2022-08-27 13:30:21",
                            "uuid": bx_job_uuid,
                            "bx_workflow_uuid": bx_workflow_uuid,
                            "label": f"job{job_number}",
                            "data_label": f"data{data_number}",
                            "state": "GOOD",
                        }
                    bx_job_dicts.append(bx_job_dict)

                    # Add a workflow for every job.
                    bx_workflow_dict = {
                        "uuid": bx_workflow_uuid,
                        "bx_job_uuid": bx_job_uuid,
                        "filename_classname": f"my/good/{bx_workflow_uuid}.py::Abc",
                    }
                    bx_workflow_dicts.append(bx_workflow_dict)

            # -----------------------------------------------------------------
            # Second record for some jobs.
            data_number = 2
            job_number = 2
            bx_job_uuid = f"job_{data_number}x{job_number}_1"
            bx_workflow_uuid = f"workflow_{data_number}x{job_number}_1"
            bx_job_dict = {
                "created_on": "2022-08-27 13:30:22",
                "uuid": bx_job_uuid,
                "bx_workflow_uuid": bx_workflow_uuid,
                "label": f"job{job_number}",
                "data_label": f"data{data_number}",
                "state": "BAD",
            }
            bx_job_dicts.append(bx_job_dict)

            bx_workflow_dict = {
                "uuid": bx_workflow_uuid,
                "bx_job_uuid": bx_job_uuid,
                "filename_classname": f"my/good/{bx_workflow_uuid}.py::Abc",
            }
            bx_workflow_dicts.append(bx_workflow_dict)

            # -----------------------------------------------------------------
            prepend_job_labels = {
                "prepended": {
                    "workflow_filename_classname": "prepended.py::Prepended",
                }
            }
            append_job_labels = {
                "appended": {
                    "workflow_filename_classname": "appended.py::Appended",
                },
                "other": {
                    "workflow_filename_classname": "other.py::Other",
                },
            }

            exclude_job_labels = ["job3", "other"]

            # -----------------------------------------------------------------
            # Add the jobs and their workflows to the database.
            await bx_datafaces_get_default().set_bx_jobs(bx_job_dicts)
            await bx_datafaces_get_default().set_bx_workflows(bx_workflow_dicts)

            # Query the records to be composed.
            # This will have JOIN between jobs and workflows to get the filename_classname.
            records = await bx_datafaces_get_default().get_bx_jobs()

            text_composer = BxComposers().build_object(
                {"type": "dls_bxflow_lib.bx_composers.text"}
            )

            text = text_composer.compose_bx_jobs_data_grid(
                records,
                prepend_job_labels=prepend_job_labels,
                append_job_labels=append_job_labels,
                exclude_job_labels=exclude_job_labels,
            )

            logger.debug(f"composed text\n{text}")

            html_composer = BxComposers().build_object(
                {"type": "dls_bxflow_lib.bx_composers.html"}
            )

            html = html_composer.compose_bx_jobs_data_grid(
                records,
                prepend_job_labels=prepend_job_labels,
                append_job_labels=append_job_labels,
                exclude_job_labels=exclude_job_labels,
            )

            logger.debug(f"composed html\n{html}")

            # -----------------------------------------------------------------
            # Grid with no records.
            records = {}

            text_composer = BxComposers().build_object(
                {"type": "dls_bxflow_lib.bx_composers.text"}
            )

            text = text_composer.compose_bx_jobs_data_grid(
                records,
                prepend_job_labels=prepend_job_labels,
                append_job_labels=append_job_labels,
                exclude_job_labels=exclude_job_labels,
            )

            logger.debug(f"composed text\n{text}")

            html_composer = BxComposers().build_object(
                {"type": "dls_bxflow_lib.bx_composers.html"}
            )

            html = html_composer.compose_bx_jobs_data_grid(
                records,
                prepend_job_labels=prepend_job_labels,
                append_job_labels=append_job_labels,
                exclude_job_labels=exclude_job_labels,
            )

            logger.debug(f"composed html\n{html}")
