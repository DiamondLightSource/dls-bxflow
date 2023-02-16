import json
import logging
import time

# Utilities.
from dls_utilpack.describe import describe
from dls_utilpack.isodatetime import isodatetime

# Base class for a bx_task.
from dls_bxflow_run.bx_tasks.base import Base

# Task type constants.
from dls_bxflow_run.bx_tasks.bx_tasks import BxTaskTypes

# States.
from dls_bxflow_run.bx_tasks.states import States

logger = logging.getLogger(__name__)

thing_type = BxTaskTypes.DUMMY


class Dummy(Base):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        Base.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.state(States.PREPARED)

    # ----------------------------------------------------------------------------------------
    async def run(self):
        """
        Dummy will read infile_variable and write outfile_variable if given.
        """

        task_label = self.specification()["label"]

        type_specific_tbd = self.specification().get("type_specific_tbd", {})

        deliberate_error = type_specific_tbd.get("deliberate_error")
        if deliberate_error is not None:
            raise RuntimeError(deliberate_error)

        infile_variable = type_specific_tbd.get("infile_variable", None)
        if infile_variable is not None:
            logger.debug(describe("infile_variable", infile_variable))

            # Find variable by name.
            # infile = self.variables.find(infile_variable, "name").trait("value")

            # dummy_bx_job_token = None
            # if infile is not None:
            #     with open(infile, "r") as stream:
            #         dummy_bx_job_token = stream.read()

        outfile_variable = type_specific_tbd.get("outfile_variable", None)

        if outfile_variable is not None:
            # Find variable by name.
            outfile = self.variables.find(outfile_variable, "name").trait("value")

        else:
            # logger.debug("no outfile_variable, looking for outfile in specification")
            outfile = type_specific_tbd.get("outfile", None)

        delay = type_specific_tbd.get("delay", 0)
        if delay > 0:
            t0 = time.time()
            while time.time() - t0 < delay:
                time.sleep(1.0)
                logger.debug(
                    "%s delayed %0.3f out of %0.3f"
                    % (self.label(), time.time() - t0, delay)
                )

        if outfile is not None:
            logger.debug("writing outfile %s" % (outfile))
            with open(outfile, "wt") as stream:
                output = {
                    "bx_task_label": task_label,
                    "datetime": isodatetime(),
                }
                json.dump(output, stream, indent=4)

            # The output filename will be a candidate for a catalog attachment.
            self.propose_artefact(outfile)

        else:
            logger.debug("no outfile, writing nothing")

        # Always write a local file.
        with open(f"{task_label}.data", "wt") as stream:
            output = {
                "bx_task_label": task_label,
                "datetime": isodatetime(),
            }
            json.dump(output, stream, indent=4)

        return 0

    # ------------------------------------------------------------------------------------------
    def extract_error_lines(self):
        """
        Get error lines from log file.
        """

        # This task expects to run in python setup using dls-logformatter.
        # Use the base-class method to get error lines from logging formatter logs.
        return self.extract_error_lines_from_dls_logformatter()
