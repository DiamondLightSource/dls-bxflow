# Use standard logging in this module.
import logging

logger = logging.getLogger()


class Overall:
    def __init__(self):
        self.__ispyb_commands = {}

    def add_ispyb_command(self, command, parser):
        if command not in self.__ispyb_commands:
            self.__ispyb_commands[command] = {"count": 0, "recipe_filenames": []}
        self.__ispyb_commands[command]["count"] += 1
        if (
            parser.recipe_filename()
            not in self.__ispyb_commands[command]["recipe_filenames"]
        ):
            self.__ispyb_commands[command]["recipe_filenames"].append(
                parser.recipe_filename()
            )

    def compose_as_prettytable(self, should_include_recipes=False):

        import prettytable

        table = prettytable.PrettyTable()
        table.field_names = [
            "command",
            "count",
        ]

        if should_include_recipes:
            table.field_names.append("recipes")

        for ispyb_command, info in self.__ispyb_commands.items():
            row = []
            row.append(ispyb_command)
            row.append(info["count"])
            if should_include_recipes:
                row.append("\n".join(info["recipe_filenames"]))
            table.add_row(row)
        table.align = "l"
        table.title = "ispyb command distribution"

        return str(table)
