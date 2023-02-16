import logging
import os

import pytest
from dls_utilpack.import_class import ImportClassFailed

# Exceptions.
from dls_bxflow_api.exceptions import DuplicateLabelException, DuplicateUuidException

# Remex (remote execution) API.
from dls_bxflow_api.remex import Clusters as RemexClusters
from dls_bxflow_api.remex import Keywords as RemexKeywords

# Context creator.
from dls_bxflow_lib.bx_contexts.bx_contexts import BxContexts

# Object managing bx_tasks.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTasks, BxTaskTypes

# Base class for the tester.
from tests.base_context_tester import BaseContextTester

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class Task:
    def __init__(self, id):
        self.__id = id

    def runit_sync(self):
        logger.debug("ran sync")
        self.__write()

    async def runit_async(self):
        logger.debug("ran async")
        self.__write()

    def __write(self):
        with open(f"{self.__id}.txt", "w") as stream:
            stream.write(f"{self.__id}\n")


# ----------------------------------------------------------------------------------------
class TestDuplicateTask:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/filestore.yaml"
        TaskDuplicateTester().main(constants, configuration_file, output_directory)


# ----------------------------------------------------------------------------------------
class TestModulenameClassnameTask:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/filestore.yaml"
        TaskModulenameClassnameTester().main(
            constants, configuration_file, output_directory
        )


# ----------------------------------------------------------------------------------------
class TestFilenameClassnameTask:
    def test(self, constants, logging_setup, output_directory):
        """ """

        configuration_file = "tests/configurations/filestore.yaml"
        TaskFilenameClassnameTester().main(
            constants, configuration_file, output_directory
        )


# ----------------------------------------------------------------------------------------
class TaskDuplicateTester(BaseContextTester):
    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()
        specification = await bx_configurator.load()
        bx_context = BxContexts().build_object(specification)

        async with bx_context:
            await self.duplicate()

    # ------------------------------------------------------------------------
    async def __build_dummy_task(self, label, predefined_uuid=None):
        """
        Build a dummy task object.
        """

        bx_task_specification = {
            "type": "dls_bxflow_run.bx_tasks.dummy",
            "label": label,
            "remex_hints": {
                RemexKeywords.CLUSTER: RemexClusters.LOCAL,
            },
        }

        # Build the task.
        bx_task = BxTasks().build_object(
            bx_task_specification, predefined_uuid=predefined_uuid
        )

        return bx_task

    # ----------------------------------------------------------------------------------------
    async def duplicate(self):
        """ """

        bx_tasks = BxTasks()

        bx_task1 = await self.__build_dummy_task("task 1")
        bx_tasks.add(bx_task1)

        with pytest.raises(DuplicateLabelException):
            bx_tasks.add(bx_task1)

        bx_task2 = await self.__build_dummy_task(
            "task 2", predefined_uuid=bx_task1.uuid()
        )

        logger.info(f"bx_task1.uuid() {bx_task1.uuid()}")
        logger.info(f"bx_task2.uuid() {bx_task2.uuid()}")
        with pytest.raises(DuplicateUuidException):
            bx_tasks.add(bx_task2)

        bx_task3 = await self.__build_dummy_task("task 3")
        bx_tasks.add(bx_task3)

        assert str(bx_tasks) == "BxTasks [task 1, task 3]"
        logger.info("bx_tasks is %s" % (bx_tasks))


# ----------------------------------------------------------------------------------------
class TaskModulenameClassnameTester(BaseContextTester):
    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()
        specification = await bx_configurator.load()
        bx_context = BxContexts().build_object(specification)

        # The work is done in the output directory.
        old_cwd = os.getcwd()
        try:
            os.chdir(output_directory)
            async with bx_context:
                await self.modulename_classname()
        finally:
            os.chdir(old_cwd)

    # ------------------------------------------------------------------------
    async def modulename_classname(self):
        """
        Test the modulename_classname task type.
        """

        bx_task_specification = {
            "type": BxTaskTypes.MODULE_CLASSNAME,
            "label": "modulename_classname",
            "remex_hints": {
                RemexKeywords.CLUSTER: RemexClusters.LOCAL,
            },
            "type_specific_tbd": {
                "module_classname": "tests.test_task::Task",
                "run_method": "runit_sync",
                "constructor_args": [1],
            },
        }

        # -----------------------------------------------------
        # Build the task as specified.
        id = "good_sync"
        bx_task_specification["type_specific_tbd"]["constructor_args"] = [id]
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        # Run it.
        await bx_task.run()
        assert os.path.exists(f"{id}.txt")

        # -----------------------------------------------------
        # Build the task for async.
        id = "good_async"
        bx_task_specification["type_specific_tbd"]["constructor_args"] = [id]
        bx_task_specification["type_specific_tbd"]["run_method"] = "runit_async"
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        # Run it.
        await bx_task.run()
        assert os.path.exists(f"{id}.txt")

        # -----------------------------------------------------
        # Build the task for bogus method.
        bx_task_specification["type_specific_tbd"]["run_method"] = "runit_bogus"
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        with pytest.raises(AttributeError) as exception_info:
            # Run it.
            await bx_task.run()

        logger.debug(str(exception_info.value))
        assert "runit_bogus" in str(exception_info.value)

        # -----------------------------------------------------
        # Build the task for bad formatted name.
        bx_task_specification["type_specific_tbd"]["module_classname"] = "bogus.bogus"
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        with pytest.raises(RuntimeError) as exception_info:
            # Run it.
            await bx_task.run()
        assert "is not of form" in str(exception_info.value)

        # -----------------------------------------------------
        # Build the task for bogus module.
        bx_task_specification["type_specific_tbd"][
            "module_classname"
        ] = "bogus.bogus::Bogus"
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        with pytest.raises(RuntimeError) as exception_info:
            # Run it.
            await bx_task.run()
        assert "No module named 'bogus'" in str(exception_info.value)

        # -----------------------------------------------------
        # Build the task for bogus class.
        bx_task_specification["type_specific_tbd"][
            "module_classname"
        ] = "tests.test_task::Bogus"
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        with pytest.raises(ImportClassFailed) as exception_info:
            # Run it.
            await bx_task.run()
        assert "could not find class Bogus" in str(exception_info.value)


# ----------------------------------------------------------------------------------------
class TaskFilenameClassnameTester(BaseContextTester):
    async def _main_coroutine(self, constants, output_directory):
        """ """

        bx_configurator = self.get_bx_configurator()
        specification = await bx_configurator.load()
        bx_context = BxContexts().build_object(specification)

        # The work is done in the output directory.
        old_cwd = os.getcwd()
        try:
            os.chdir(output_directory)
            async with bx_context:
                await self.filename_classname()
        finally:
            os.chdir(old_cwd)

    # ------------------------------------------------------------------------
    async def filename_classname(self):
        """
        Test the filename_classname task type.
        """

        bx_task_specification = {
            "type": BxTaskTypes.FILENAME_CLASSNAME,
            "label": "filename_classname",
            "remex_hints": {
                RemexKeywords.CLUSTER: RemexClusters.LOCAL,
            },
            "type_specific_tbd": {
                "filename_classname": f"{__file__}::Task",
                "run_method": "runit_sync",
                "constructor_args": [1],
            },
        }

        # -----------------------------------------------------
        # Build the task as specified.
        id = "good_sync"
        bx_task_specification["type_specific_tbd"]["constructor_args"] = [id]
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        # Run it.
        await bx_task.run()
        assert os.path.exists(f"{id}.txt")

        # -----------------------------------------------------
        # Build the task for async.
        id = "good_async"
        bx_task_specification["type_specific_tbd"]["constructor_args"] = [id]
        bx_task_specification["type_specific_tbd"]["run_method"] = "runit_async"
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        # Run it.
        await bx_task.run()
        assert os.path.exists(f"{id}.txt")

        # -----------------------------------------------------
        # Build the task for bogus method.
        bx_task_specification["type_specific_tbd"]["run_method"] = "runit_bogus"
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        with pytest.raises(AttributeError) as exception_info:
            # Run it.
            await bx_task.run()

        logger.debug(str(exception_info.value))
        assert "runit_bogus" in str(exception_info.value)

        # -----------------------------------------------------
        # Build the task for bad formatted name.
        bx_task_specification["type_specific_tbd"][
            "filename_classname"
        ] = "bogus/bogus.py"
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        with pytest.raises(RuntimeError) as exception_info:
            # Run it.
            await bx_task.run()
        assert "is not of form" in str(exception_info.value)

        # -----------------------------------------------------
        # Build the task for bogus module.
        bx_task_specification["type_specific_tbd"][
            "filename_classname"
        ] = "bogus/bogus.py::Bogus"
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        with pytest.raises(RuntimeError) as exception_info:
            # Run it.
            await bx_task.run()
        assert "could not find python file bogus/bogus.py" in str(exception_info.value)

        # -----------------------------------------------------
        # Build the task for bogus class.
        bx_task_specification["type_specific_tbd"][
            "filename_classname"
        ] = f"{__file__}::Bogus"
        bx_task = BxTasks().build_object(
            bx_task_specification,
        )

        with pytest.raises(ImportClassFailed) as exception_info:
            # Run it.
            await bx_task.run()
        assert "could not find class Bogus" in str(exception_info.value)
