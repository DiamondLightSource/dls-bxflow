import json
import logging
import os

# Utilities.
from dls_utilpack.callsign import callsign
from dls_utilpack.require import require

# Base class for a dawn task.
from dls_bxflow_run.bx_tasks.dawn_base import DawnBase

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_run.bx_tasks.dawn2"


class Dawn2(DawnBase):
    """
    Task which runs DAWN, given a DAWN json template and an input data filename.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None, predefined_uuid=None):
        DawnBase.__init__(
            self, thing_type, specification, predefined_uuid=predefined_uuid
        )

    # ----------------------------------------------------------------------------------------
    async def run(self):
        """ """

        # Get the required specification fields.
        type_specific_tbd = self.specification().get("type_specific_tbd", {})
        template_filename = require(
            f"{callsign(self)} configuration",
            type_specific_tbd,
            "template_filename",
        )
        data_filename = require(
            f"{callsign(self)} configuration",
            type_specific_tbd,
            "data_filename",
        )

        # Read the json template given in the specification.
        stream = None
        try:
            stream = open(template_filename, "r")
            try:
                template = json.load(stream)
            except Exception:
                raise RuntimeError(
                    f"unable to parse contents of DAWN configuration file {template_filename}"
                )
        except Exception:
            raise RuntimeError(
                f"unable to open DAWN configuration file {template_filename}"
            )
        finally:
            if stream is not None:
                stream.close()

        # Get the template's name without directory or suffix.
        # DON'T CHANGE THIS UNLESS ADDRESSING THE COMPANION NOTEBOOK TASK!
        template_name = os.path.splitext(os.path.basename(template_filename))[0]

        # The output filename will be a candidate for a catalog attachment.
        runtime_directory = os.getcwd()
        output_filename = f"{runtime_directory}/{template_name}.nxs"
        self.propose_artefact(output_filename)

        # Replace fields in the template.
        template["runDirectory"] = runtime_directory
        template["filePath"] = data_filename
        template["outputFilePath"] = output_filename

        # Write the now-populated template as json file into the current directory.
        cwd_template_filename = f"{template_name}.json"
        with open(cwd_template_filename, "w") as stream:
            json.dump(template, stream, indent=4)

        exit_code = self.launch_and_wait(cwd_template_filename)

        # TODO: In bx_task dawn2 (and others) investigate if run command should return anything.
        return exit_code
