# Use standard logging in this module.
import logging

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------
__default_filestore = None


def bx_filestores_set_default(filestore):
    global __default_filestore
    __default_filestore = filestore


def bx_filestores_get_default():
    global __default_filestore
    if __default_filestore is None:
        raise RuntimeError("bx_filestores_get_default instance is None")
    return __default_filestore


# -----------------------------------------------------------------------------------------


class BxFilestores(Things):
    """
    List of available bx_filestores.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification):
        """"""

        filestore_class = self.lookup_class(specification["type"])

        try:
            filestore_object = filestore_class(specification)
        except Exception as exception:
            raise RuntimeError(
                "unable to build filestore object for type %s" % (filestore_class)
            ) from exception

        return filestore_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if class_type == "dls_bxflow_lib.bx_filestores.explicit":
            from dls_bxflow_lib.bx_filestores.explicit import Explicit

            return Explicit

        if class_type == "dls_bxflow_lib.bx_filestores.scandir":
            from dls_bxflow_lib.bx_filestores.scandir import Scandir

            return Scandir

        else:
            try:
                RuntimeClass = Things.lookup_class(self, class_type)
                return RuntimeClass
            except NotFound:
                raise NotFound(
                    "unable to get filestore class for type %s" % (class_type)
                )
