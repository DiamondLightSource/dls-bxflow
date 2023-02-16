import json

# Use standard logging in this module.
import logging

# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)


class BxGates(Things):
    """
    List of available bx_gates.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, label="bx_gates"):
        Things.__init__(self, label)

    # -----------------------------------------------------------------------------
    async def find_by_label(self, label):
        for bx_gate in self.list():
            if bx_gate.label() == label:
                return bx_gate
        raise NotFound("no bx_gate labeld %s" % (label))

    # -----------------------------------------------------------------------------
    async def register(self, bx_job_uuid, bx_task_uuid):

        bx_gates_list = []
        for bx_gate in self.list():
            bx_gate_dict = {}
            bx_gate_dict["bx_job_uuid"] = bx_job_uuid
            bx_gate_dict["bx_task_uuid"] = bx_task_uuid
            bx_gate_dict["type"] = bx_gate.thing_type()
            bx_gate_dict["uuid"] = bx_gate.uuid()
            bx_gate_dict["state"] = bx_gate.state()
            bx_gate_dict["specification"] = json.dumps(bx_gate.specification())
            # Breaks pattern to put something which originated
            # in the specification as separate database field.
            bx_gate_dict["label"] = bx_gate.label()
            bx_gates_list.append(bx_gate_dict)

        await bx_datafaces_get_default().set_bx_gates(bx_gates_list)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification=None, predefined_uuid=None):
        """"""

        if specification is None:
            specification = {"type": "dls_bxflow_lib.bx_gates.standard"}

        # If a string, parse for json, yaml or whatever.
        specification = self.parse_specification(specification)

        bx_gate_class = self.lookup_class(
            specification["type"],
        )

        try:
            bx_gate_object = bx_gate_class(
                specification, predefined_uuid=predefined_uuid
            )
        except Exception as exception:
            raise RuntimeError(
                "unable to build bx_gate object for type %s" % (bx_gate_class)
            ) from exception

        return bx_gate_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_gates.standard":
            from dls_bxflow_run.bx_gates.standard import Standard

            return Standard

        raise NotFound("unable to get bx_gate class for type %s" % (class_type))
