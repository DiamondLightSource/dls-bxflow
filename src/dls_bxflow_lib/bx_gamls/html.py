import logging

import yaml

# Base class for gaml things.
from dls_bxflow_lib.bx_gamls.base import Base as BxGamlBase

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_gamls.bare"


class Html(BxGamlBase):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        BxGamlBase.__init__(self, thing_type, specification)

        self.__lines = None
        self.__indent = 0

    # ----------------------------------------------------------------------------------------
    async def load(self, filename):
        try:
            with open(filename, "r") as yaml_stream:
                loaded_string = yaml_stream.read()
        except Exception as exception:
            raise RuntimeError(f"unable to read {filename}") from exception

        try:
            loaded_dict = yaml.safe_load(loaded_string)
        except Exception as exception:
            raise RuntimeError(f"unable to parse {filename}") from exception

        return loaded_dict

    # ----------------------------------------------------------------------------------------
    def parse(self, payload):
        """
        Parse a payload, such as is returned from a html ajax call.
        """

        contents = {}
        for field, value in payload.items():
            parts = field.split(".")
            leaf = contents
            for part in parts[:-1]:
                if part not in leaf:
                    leaf[part] = {}
                leaf = leaf[part]
            leaf[parts[-1]] = value

        return contents

    # ----------------------------------------------------------------------------------------
    def save(self, contents, filename):
        pass

    # ----------------------------------------------------------------------------------------
    def compose(self, contents):
        """
        Compose the contents into input fields suitable for display.
        """
        self.__lines = []
        self._compose_div("", contents, "")

        return "\n".join(self.__lines)

    # ----------------------------------------------------------------------------------------
    def _compose_div(self, key, contents, name_chain):
        """
        Compose an HTML div, recursive.
        """
        prefix = " " * self.__indent
        self.__lines.append(f"{prefix}<div class='T_section'>")
        self.__indent += 2
        prefix = " " * self.__indent

        self.__lines.append(f"{prefix}<div class='T_title'>{key}</div>")
        self.__lines.append(f"{prefix}<div class='T_body'>")

        for key, content in contents.items():
            if isinstance(content, dict):
                if name_chain != "":
                    name_chain += "."
                name_chain += key
                self.__indent += 2
                self._compose_div(key, content, name_chain)
                self.__indent -= 2
            else:
                self._compose_input(key, content, name_chain)

        self.__lines.append(f"{prefix}</div><!-- T_body -->")

        self.__indent -= 2
        prefix = " " * self.__indent
        self.__lines.append(f"{prefix}</div><!-- T_section -->")

    # ----------------------------------------------------------------------------------------
    def _compose_input(self, key, value, name_chain):
        """
        Componse an HTML input field.
        """
        self.__indent += 2
        prefix1 = " " * self.__indent
        prefix2 = " " * (self.__indent + 2)
        self.__lines.append(f"{prefix1}<div class='T_input'>")
        self.__lines.append(f"{prefix2}<div class='T_prompt'>{key}</div>")
        if name_chain == "":
            name = key
        else:
            name = f"{name_chain}.{key}"
        self.__lines.append(
            f"{prefix2}<div class='T_value'><input type='TEXT' name='{name}' value='{value}' /></div>"
        )
        self.__lines.append(f"{prefix1}</div>")
        self.__indent -= 2
