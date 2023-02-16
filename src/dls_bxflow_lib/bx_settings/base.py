import html
import logging

# Utilities.
from dls_utilpack.describe import describe

# Base class for generic things.
from dls_bxflow_api.thing import Thing

logger = logging.getLogger(__name__)


class Base(Thing):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, thing_type, specification=None, predefined_uuid=None):

        # uuid = require("%s specification" % (thing_type), specification, "uuid")

        Thing.__init__(self, thing_type, specification, predefined_uuid=predefined_uuid)

        self.__prompt = self.specification().get("prompt", self.uuid())

        self.__explanation_lines = self.specification().get("explanation", [])
        if not isinstance(self.__explanation_lines, list):
            self.__explanation_lines = [self.__explanation_lines]

        value = self.specification().get("value")

        logger.debug(
            describe(f"[SETTINGVAL] constructing setting {self.uuid()}", value)
        )

        self.set_value(value)

    # ----------------------------------------------------------------------------------------
    def get_value(self):
        """"""
        return self.__value

    # ----------------------------------------------------------------------------------------
    def set_value(self, value):
        """"""
        self.__value = value

    # ----------------------------------------------------------------------------------------
    def as_dict(self):
        """"""

        dict = {}
        dict["uuid"] = self.uuid()
        dict["value"] = self.encode_for_saving()
        dict["type"] = self.thing_type()

        dict["prompt"] = self.__prompt

        return dict

    # ----------------------------------------------------------------------------------------
    def compose_html_input(self):
        """"""

        html_lines = []
        html_lines.append("<tr class='T_setting'>")

        html_lines.append(f"    <td><div class='T_prompt'>{self.__prompt}</div></td>")

        if self.__value is None:
            escaped_value = ""
        else:
            escaped_value = html.escape(str(self.__value))

        html_lines.append("    <td>")
        html_lines.append(
            f"    <div class='T_input {self.css_style}'><input type='text'"
            f" name='{self.uuid()}'"
            f" value='{escaped_value}' />"
            "</div>"
        )

        html_lines.append("    <div class='T_explanation_lines'>")

        for explanation_line in self.__explanation_lines:
            html_lines.append(f"      <div>{explanation_line}</div>")

        html_lines.append("    </div>")
        html_lines.append("  </td>")

        html_lines.append("</tr>")

        return "\n".join(html_lines)

    # ----------------------------------------------------------------------------------------
    def compose_html_output(self):
        """"""

        html_lines = []
        html_lines.append("<tr>")

        html_lines.append(f"    <td class='T_prompt'>{self.__prompt}</td>")

        if self.__value is None:
            escaped_value = ""
        else:
            escaped_value = html.escape(str(self.__value))

        html_lines.append(
            f"    <td class='T_output {self.css_style}'>{escaped_value}</td>"
        )

        html_lines.append("</tr>")

        return "\n".join(html_lines)

    # ----------------------------------------------------------------------------------------
    def compose_python_variable_assignment(self):
        """
        Return a list of one or more lines that accomplishes a python assignment.
        TODO: Need type-specific overrides for setting.compose_python_variable_assignment.
        """

        python_lines = []
        if self.__prompt != self.uuid():
            python_lines.append(f"# {self.__prompt}")

        python_lines.append(f"{self.uuid()} = {self.encode_for_python()}")

        return python_lines

    # ----------------------------------------------------------------------------------------
    def encode_for_python(self):
        """"""

        return repr(self.get_value())

    # ----------------------------------------------------------------------------------------
    def encode_for_saving(self):
        """"""

        return self.get_value()

    # ----------------------------------------------------------------------------------------
    def add_to_argument_parser(self, parser):
        """"""

        metavar = f"{self.uuid()} ({self.get_type_name()}, default {self.get_value()})"

        logger.debug(f"[METAVAR] {metavar}")

        parser.add_argument(
            f"--{self.uuid()}",
            # type=self.type_converter,
            dest=self.uuid(),
            metavar=metavar,
            required=False,
            default=self.get_value(),
        )
