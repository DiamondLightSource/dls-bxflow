import logging

import yaml

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------


class BxContexts(Things):
    """
    Context loader.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification):
        """"""

        if not isinstance(specification, dict):
            with open(specification, "r") as yaml_stream:
                specification = yaml.safe_load(yaml_stream)

        bx_context_class = self.lookup_class(specification["type"])

        try:
            bx_context_object = bx_context_class(specification)
        except Exception as exception:
            raise RuntimeError(
                "unable to build bx_context object for type %s" % (bx_context_class)
            ) from exception

        return bx_context_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_contexts.classic":
            from dls_bxflow_lib.bx_contexts.classic import Classic

            return Classic

        raise NotFound("unable to get bx_context class for type %s" % (class_type))
