import json

# Use standard logging in this module.
import logging

import yaml

# Utilities.
from dls_utilpack.callsign import callsign, callsign_html
from dls_utilpack.explain import explain
from dls_utilpack.require import require

# Exceptions.
from dls_bxflow_api.exceptions import NotFound

# Class managing list of things.
from dls_bxflow_api.things import Things

# Setting types.
from dls_bxflow_lib.bx_settings.constants import Types as SettingTypes

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------


class BxSettings(Things):
    """
    List of available bx_settings.
    """

    # ----------------------------------------------------------------------------------------
    def __init__(self, name=None):
        Things.__init__(self, name)

    # ----------------------------------------------------------------------------------------
    def callsign(self):
        """"""

        name = self.name()

        if name is not None:
            return f"BxSettings for {name}"
        else:
            return Things.callsign()

    # ----------------------------------------------------------------------------------------
    def build_object(self, specification, predefined_uuid=None):
        """"""

        if predefined_uuid is None and "uuid" in specification:
            predefined_uuid = specification["uuid"]

        setting_type = specification.get("type", SettingTypes.STRING)

        setting_class = self.lookup_class(setting_type)

        try:
            setting_object = setting_class(
                specification, predefined_uuid=predefined_uuid
            )
        except Exception as exception:
            raise RuntimeError(
                "unable to build setting object for type %s" % (setting_class)
            ) from exception

        return setting_object

    # ----------------------------------------------------------------------------------------
    def lookup_class(self, class_type):
        """"""

        if (
            class_type == SettingTypes.BOOLEAN
            or class_type == "boolean"
            or class_type == "bool"
        ):
            from dls_bxflow_lib.bx_settings.boolean import Boolean

            return Boolean

        if (
            class_type == SettingTypes.STRING
            or class_type == "string"
            or class_type == "str"
        ):
            from dls_bxflow_lib.bx_settings.string import String

            return String

        if class_type == SettingTypes.FLOAT or class_type == "float":
            from dls_bxflow_lib.bx_settings.float import Float

            return Float

        if (
            class_type == SettingTypes.INTEGER
            or class_type == "integer"
            or class_type == "int"
        ):
            from dls_bxflow_lib.bx_settings.integer import Integer

            return Integer

        raise NotFound("unable to get setting class for type %s" % (class_type))

    # ----------------------------------------------------------------------------------------
    def get_value(self, uuid):
        """"""
        try:
            return self.find(uuid).get_value()
        except NotFound:
            raise NotFound(f"{self.name()} does not have setting {uuid}")

    # ----------------------------------------------------------------------------------------
    def get_value_or_default(self, uuid, default_value=None):
        """"""
        try:
            return self.find(uuid).get_value()
        except NotFound:
            return default_value

    # ----------------------------------------------------------------------------------------
    def set_value(self, uuid, value):
        """"""
        try:
            self.find(uuid).set_value(value)
        except NotFound:
            raise NotFound(f"{self.name()} does not have setting {uuid}")

    # ----------------------------------------------------------------------------------------
    def as_list_of_dicts(self):
        """"""

        setting_dicts = []
        for setting in self.list():
            setting_dicts.append(setting.as_dict())

        return setting_dicts

    # ----------------------------------------------------------------------------------------
    def as_dict(self):
        """"""

        kvp = {}
        # Make a dict of just the keywords and their values.
        for setting in self.list():
            kvp[setting.uuid()] = setting.get_value()

        return kvp

    # ----------------------------------------------------------------------------------------
    def load_from_json(self, json_string):
        """
        Build the interal settings list from a json string.
        Typically this json_string comes out of a string database field.
        """

        if json_string is None or json_string == "":
            return

        try:
            settings_dicts = json.loads(json_string)

            self.load_from_dicts(settings_dicts)
        except Exception as exception:
            raise RuntimeError(
                explain(
                    exception, f"{callsign(self)} loading settings from json string"
                )
            )

    # ----------------------------------------------------------------------------------------
    def load_from_dicts(self, setting_dicts, should_clear_first=True):
        """"""

        if should_clear_first:
            self.clear()

        try:
            if isinstance(setting_dicts, list):
                for setting_dict in setting_dicts:
                    self.add(self.build_object(setting_dict))
            elif isinstance(setting_dicts, dict):
                for uuid, setting_dict in setting_dicts.items():
                    self.add(self.build_object(setting_dict, predefined_uuid=uuid))
        except Exception as exception:
            raise (explain(exception, f"{callsign(self)} building settings from dicts"))

    # ----------------------------------------------------------------------------------------
    def save(self, filename):
        """"""

        try:
            with open(filename, "w") as yaml_stream:
                yaml.safe_dump(
                    {"settings": self.as_list_of_dicts()},
                    yaml_stream,
                    default_flow_style=False,
                )
        except Exception as exception:
            raise RuntimeError(
                explain(
                    exception, f"{callsign(self)} saving settings to file {filename}"
                )
            )

    # ----------------------------------------------------------------------------------------
    def load(self, filename):
        try:
            with open(filename, "r") as yaml_stream:
                loaded_string = yaml_stream.read()

            loaded_dict = yaml.safe_load(loaded_string)

            setting_dicts = require(filename, loaded_dict, "settings")

            self.load_from_dicts(setting_dicts)

        except Exception as exception:
            raise RuntimeError(
                explain(
                    exception, f"{callsign(self)} loading settings from file {filename}"
                )
            )

    # ----------------------------------------------------------------------------------------
    def update_values(self, those_bx_settings):
        """
        Update current settings from other settings.
        Creates a new settings as necessary.
        """

        for that_setting in those_bx_settings.list():
            if not self.has(that_setting.uuid()):
                self.add(that_setting)
            else:
                this_setting = self.find(that_setting.uuid())
                this_setting.set_value(that_setting.get_value())

    # ----------------------------------------------------------------------------------------
    def compose_html_inputs(self, exclusions=None):
        """"""

        setting_lines = []
        for setting in self.list():
            if exclusions is None or setting.uuid() not in exclusions:
                setting_lines.append(setting.compose_html_input())

        html_lines = []
        html_lines.append(callsign_html(self))
        if len(setting_lines) > 0:
            html_lines.append("<table class='T_settings'>")
            html_lines.extend(setting_lines)
            html_lines.append("</table><!-- T_settings -->")
        else:
            html_lines.append("<div class='T_settings'>")
            html_lines.append("There are no defined settings.")
            html_lines.append("</div><!-- T_settings -->")

        return "\n".join(html_lines)

    # ----------------------------------------------------------------------------------------
    def compose_html_outputs(self, exclusions=None):
        """"""

        setting_lines = []
        for setting in self.list():
            if exclusions is None or setting.uuid() not in exclusions:
                setting_lines.append(setting.compose_html_output())

        html_lines = []
        html_lines.append(callsign_html(self))
        if len(setting_lines) > 0:
            html_lines.append("<table class='T_settings'>")
            html_lines.append("<tr><th>name</th><th>value</th></tr>")
            html_lines.extend(setting_lines)
            html_lines.append("</table><!-- T_settings -->")
        else:
            html_lines.append("<div class='T_settings'>")
            html_lines.append("There are no defined settings.")
            html_lines.append("</div><!-- T_settings -->")

        return "\n".join(html_lines)

    # ----------------------------------------------------------------------------------------
    def compose_python_dict_assignment(self, assign_to):
        """"""

        python_lines = []
        python_lines.append("%s = {" % (assign_to))

        # Make a dict of just the keywords and their values.
        for setting in self.list():
            python_lines.append(
                '  "%s": %s,' % (setting.uuid(), setting.encode_for_python())
            )

        python_lines.append("}")

        return "\n".join(python_lines)

    # ----------------------------------------------------------------------------------------
    def compose_python_variable_assignment(self):
        """"""

        python_lines = []
        for setting in self.list():
            # Make a set of lines whihc accomplish a python assignment statement.
            python_lines.extend(setting.compose_python_variable_assignment())

        return python_lines

    # ----------------------------------------------------------------------------------------
    def add_to_argument_parser(self, parser):
        """"""

        for setting in self.list():
            # Let each setting become a command line argument.
            setting.add_to_argument_parser(parser)
