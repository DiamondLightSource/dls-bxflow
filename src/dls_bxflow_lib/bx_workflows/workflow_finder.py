import logging

# Utilities.
from dls_utilpack.import_class import (
    import_classname_from_filename,
    import_classname_from_modulename,
)

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Configurator.
from dls_bxflow_lib.bx_configurators.bx_configurators import (
    bx_configurators_get_default,
    bx_configurators_has_default,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------
class WorkflowFinder:
    """
    Helps find and instantiate workflows.
    """

    def __init__(self):
        self.__search_messages = []

    # ------------------------------------------------------------------
    def find_class_object(self, workflow_name):
        """
        Search various places for the name and return class object.
        """

        try:
            # Try to import it from the name directly.
            class_object = self.import_filename_classname(workflow_name)
            return self.found(class_object)
        except NotFound:
            pass

        filename_classname = None
        if filename_classname is None:
            # Try to find the name in the configurator.
            filename_classname = self.find_from_configurator(workflow_name)

        if filename_classname is not None:
            try:
                # From the name, try to import the class.
                class_object = self.import_filename_classname(filename_classname)
                return self.found(class_object)
            except NotFound:
                pass

        raise NotFound(self.compose_messages_as_text(workflow_name))

    # ------------------------------------------------------------------
    def found(self, class_object):
        """
        Return what was found, perhaps with some debug.
        """
        return class_object

    # ------------------------------------------------------------------
    def import_filename_classname(self, filename_classname):
        """ """

        parts = filename_classname.split("::")
        filename = parts[0]
        classnames = []

        # The workflow is not already set up with class in it?
        if len(parts) == 1:
            classnames.append("Workflow")
        else:
            classnames.append(parts[1])

        for classname in classnames:
            try:
                class_object = import_classname_from_filename(classname, filename)
                self.add_message(f"got class {classname} from file {filename}")
                return class_object
            except Exception as exception:
                self.add_message(str(exception))

            try:
                class_object = import_classname_from_modulename(classname, filename)
                self.add_message(f"got class {classname} from module {filename}")
                return class_object
            except Exception as exception:
                self.add_message(str(exception))

        raise NotFound(f"unable to import {filename_classname}")

    # ------------------------------------------------------------------
    def find_from_configurator(self, workflow_name):
        """
        Find the workflow name in the configurator.
        """

        if not bx_configurators_has_default():
            self.add_message("there is no configurator")
            return None

        configurator = bx_configurators_get_default()
        keys = ["append_job_labels", "prepend_job_labels"]
        for key in keys:
            key = f"gui.job_data_grid.{key}.{workflow_name}.workflow_filename_classname"
            try:
                filename_classname = configurator.require(key)
                self.add_message(
                    f"{workflow_name} configurated for gui to be {filename_classname}"
                )
                return filename_classname
            except Exception as exception:
                self.add_message(str(exception))

        return None

    # ------------------------------------------------------------------
    def add_message(self, message):
        self.__search_messages.append(message)

    # ------------------------------------------------------------------
    def compose_messages_as_text(self, workflow_name):
        """
        Compose search messages as multi-line string for log or text display.
        """
        return "searching for workflow %s\n    %s" % (
            workflow_name,
            "\n    ".join(self.__search_messages),
        )
