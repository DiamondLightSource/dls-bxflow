# Use standard logging in this module.
import logging

# Global bx_dataface.
from dls_bxflow_api.bx_datafaces.bx_datafaces import bx_datafaces_get_default

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

# States.
from dls_bxflow_run.bx_variables.states import States

logger = logging.getLogger(__name__)


class BxVariables(Things):
    """
    List of available variables.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name="variables"):
        Things.__init__(self, name)

        # self.__bx_scheduler = bx_schedulers_get_default()

    # -----------------------------------------------------------------------------
    async def register(self, bx_job_uuid):

        variables_list = []
        for variable in self.list():
            variable_dict = {}
            variable_dict["bx_job_uuid"] = bx_job_uuid
            variable_dict["type"] = variable.thing_type()
            variable_dict["uuid"] = variable.uuid()
            variable_dict["state"] = variable.state()
            variable_dict["name"] = variable.trait("name")
            variable_dict["value"] = variable.trait("value")
            variables_list.append(variable_dict)

        await bx_datafaces_get_default().set_bx_variables(variables_list)

    # -----------------------------------------------------------------------------
    def add(self, name, value=None):
        thing = self.build_object()
        thing.traits()["name"] = name
        thing.traits()["value"] = value
        if value is None:
            thing.state(States.UNSET)
        else:
            thing.state(States.SET)
        Things.add(self, thing)

    # ----------------------------------------------------------------------------------------
    async def fetch(self, bx_job_uuid):
        """ """
        self.clear()

        # Get all variables related to current bx_job.
        records = await bx_datafaces_get_default().get_bx_variables(bx_job_uuid)

        for record in records:
            self.add(record["name"], record["value"])

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification=None):
        """"""

        if specification is None:
            specification = {"type": "dls_bxflow_lib.variables.simple"}

        variable_class = self.lookup_class(specification["type"])

        try:
            variable_object = variable_class(specification)
        except Exception as exception:
            raise RuntimeError(
                "unable to build variable object for type %s" % (type)
            ) from exception

        return variable_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.variables.simple":
            from dls_bxflow_run.bx_variables.simple import Simple

            return Simple

        raise NotFound("unable to get variable class for type %s" % (class_type))
