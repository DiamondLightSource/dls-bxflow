import inspect
import logging

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.explain import explain
from dls_utilpack.import_class import import_class
from dls_utilpack.require import require

# Base class for a bx_task.
from dls_bxflow_run.bx_tasks.base import Base

# Task types.
from dls_bxflow_run.bx_tasks.constants import Types as BxTaskTypes

# States.
from dls_bxflow_run.bx_tasks.states import States

logger = logging.getLogger(__name__)

thing_type = BxTaskTypes.FILENAME_CLASSNAME


class FilenameClassname(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.state(States.PREPARED)

    # ----------------------------------------------------------------------------------------
    async def run(self):
        """
        Instantiate the class object.
        """

        t = require(
            f"{thing_type} specification",
            self.specification(),
            "type_specific_tbd",
        )

        filename_classname = require(
            f"{thing_type} specification",
            t,
            "filename_classname",
        )

        constructor_args = t.get("constructor_args", [])
        constructor_kwargs = t.get("constructor_kwargs", {})

        # As an extra goodie, provide the class with a reference to the bx_task that is running it.
        if "bx_task" in constructor_args:
            constructor_kwargs["bx_task"] = self

        logger.debug(f"[FNCNRUN] importing task class from {filename_classname}")

        class_object = import_class(filename_classname)

        logger.debug("[FNCNRUN] constructing task object instance")

        # Construct the workflow instance.
        class_instance = class_object(*constructor_args, **constructor_kwargs)

        run_method = t.get("run_method", "run")

        run_callable = getattr(class_instance, run_method)

        try:
            if inspect.iscoroutinefunction(run_callable):
                logger.debug(
                    f"[FNCNRUN] running task object instance async method {run_method}()"
                )
                returncode = await run_callable()
            else:
                logger.debug(
                    f"[FNCNRUN] running task object instance method {run_method}()"
                )
                returncode = run_callable()
        except Exception as exception:
            raise RuntimeError(
                explain(exception, f"running {callsign(class_instance)} instance")
            )

        if returncode is None:
            returncode = 0

        return returncode

    # ------------------------------------------------------------------------------------------
    def extract_error_lines(self):
        """
        Get error lines from log file.
        """

        # This task expects to run in python setup using dls-logformatter.
        # Use the base-class method to get error lines from logging formatter logs.
        return self.extract_error_lines_from_dls_logformatter()
