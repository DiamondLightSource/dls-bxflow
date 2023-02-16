import json

# Use standard logging in this module.
import logging

# Utilities.
from dls_utilpack.require import require

logger = logging.getLogger()


class Parser1:
    def __init__(self, overall, recipe_filename):
        self.__overall = overall
        self.__recipe_filename = recipe_filename
        self.__recipe_object = {}
        self.__step_rows = []
        self.__ispyb_commands = []

    def recipe_filename(self):
        return self.__recipe_filename

    def parse(self):
        with open(self.__recipe_filename, "r") as json_stream:
            self.__recipe_object = json.load(json_stream)

        for step_number, step_object in self.__recipe_object.items():
            if step_number == "start":
                continue
            step_row = []
            step_row.append(step_number)
            queue = require("step {step_number}", step_object, "queue")
            step_row.append(queue)
            if queue == "ispyb_connector":
                variables = require("step {step_number}", step_object, "variables")
                command = require(
                    "step {step_number} variables", variables, "ispyb_command"
                )
                if command == "multipart_message":
                    command_list = []
                    for commands in variables.get("ispyb_command_list", []):
                        command = commands.get("ispyb_command", "--")
                        command_list.append(command)
                        self.__overall.add_ispyb_command(command, self)
                    command = ", ".join(command_list)
                else:
                    self.__overall.add_ispyb_command(command, self)
                step_row.append(command)
            else:
                step_row.append("-")

            self.__step_rows.append(step_row)

    def compose_as_prettytable(self):

        import prettytable

        table = prettytable.PrettyTable()
        table.field_names = [
            "step",
            "queue",
            "details",
        ]

        table.add_rows(self.__step_rows)
        table.align = "l"
        table.title = self.__recipe_filename

        return str(table)
