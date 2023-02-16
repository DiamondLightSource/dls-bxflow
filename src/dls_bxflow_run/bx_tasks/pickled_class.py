import json
import logging

import jsonpickle

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.describe import describe
from dls_utilpack.explain import explain
from dls_utilpack.require import require

# Base class for a bx_task.
from dls_bxflow_run.bx_tasks.base import Base

# States.
from dls_bxflow_run.bx_tasks.states import States

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_run.bx_tasks.pickled_class"


class PickledClass(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        type_specific_tbd = require(
            f"{thing_type} specification", self.specification(), "type_specific_tbd"
        )
        uclass = require(f"{thing_type} spefication class", type_specific_tbd, "class")

        # Specification contains a uclass which is not yet pickled?
        if not isinstance(uclass, dict):
            uclass = jsonpickle.encode(uclass)
            uclass = json.loads(uclass)

        type_specific_tbd["class"] = uclass
        self.state(States.PREPARED)

    # ----------------------------------------------------------------------------------------
    def instantiate(self):
        """
        Instantiate the class object.
        isolate:
        create directory /tmp/bx_flow/isolation/job_uuid/task_uuid/
        write configuration.json there
        cd to there and launch isolation.py with pipe to stdout.txt, stderr.txt
        isolation.py sets up logging to main.log
        the unpickles, instantiates and runs
        """

        try:
            type_specific_tbd = self.specification().get("type_specific_tbd", {})
            uclass = require("bx_task specification[class]", type_specific_tbd, "class")
            constructor_args = type_specific_tbd.get("constructor_args", [])
            constructor_kwargs = type_specific_tbd.get("constructor_kwargs", {})

        except Exception as exception:
            raise RuntimeError(
                explain(exception, f"parsing {thing_type} specification")
            )

        try:
            pickled_uclass = json.dumps(uclass)
            uclass = jsonpickle.decode(pickled_uclass)
        except Exception as exception:
            raise RuntimeError(
                explain(exception, describe("unpickling class which", uclass))
            )

        if not isinstance(uclass, type):
            raise RuntimeError(f"did not get a class decoding pickle {pickled_uclass}")

        # As an extra goodie, provide the pickled class with a reference to the bx_task that is running it.
        constructor_kwargs["bx_task"] = self

        try:
            uobject = uclass(*constructor_args, **constructor_kwargs)
        except Exception as exception:
            raise RuntimeError(explain(exception, f"instantiating {str(uclass)} class"))

        return uobject

    # ----------------------------------------------------------------------------------------
    async def run(self):
        """
        Unpickle the object and instantiate an instance.
        """

        uobject = self.instantiate()

        try:
            returncode = await uobject.run()
        except Exception as exception:
            raise RuntimeError(
                explain(exception, f"running {callsign(uobject)} instance")
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
