import html
import logging

# Database field names.
from dls_bxflow_api.bx_databases.constants import BxJobFieldnames

# Base class for generic things.
from dls_bxflow_api.thing import Thing

# Class to do the work using prettytable.
from dls_bxflow_lib.bx_composers.prettyhelper import PrettyHelper

# Filestore manager.
from dls_bxflow_lib.bx_filestores.bx_filestores import bx_filestores_get_default

# Job constants.
# TODO: Fix hierarchy crossover where dataface imports job constants.
from dls_bxflow_lib.bx_jobs.states import States as BxJobStates

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_composers.html"


class Html(Thing):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        Thing.__init__(self, thing_type, specification)

        self.__prettyhelper = PrettyHelper()

        self.__indent = 0

    # ----------------------------------------------------------------------------------------
    def compose_bx_job_details(self, bx_job_record, bx_jobs_bx_tasks_bx_gates_records):
        """"""

        # TODO: Use filestore object to discover all available files in task runtime directory.
        field_names = [
            "task",
            "task state",
            "has opened",
            "waiting for",
            "execution outputs",
        ]

        html_lines = []

        html_lines.append("<table>")
        html_lines.append("<thead>")
        html_lines.append("<tr>")
        for field_name in field_names:
            html_lines.append(f"<th>{field_name}</th>")
        html_lines.append("</tr>")
        html_lines.append("</thead>")

        rows = {}
        for record in bx_jobs_bx_tasks_bx_gates_records:
            if rows.get(record["bx_task_uuid"]) is None:
                execution_outputs = self.__render_execution_outputs(
                    record["bx_task_directory"],
                    record["bx_task_error_lines"],
                )

                rows[record["bx_task_uuid"]] = {
                    "job": record["bx_job_label"],
                    "task": record["bx_task_label"],
                    "task_state": record["bx_task_state"],
                    "controlled_bx_gates": [],
                    "dependency_bx_gates": [],
                    "execution_outputs": execution_outputs,
                }

            row = rows[record["bx_task_uuid"]]
            if record.get("controlled_bx_gate_label") is not None:
                if record["controlled_bx_gate_state"] == "open":
                    if (
                        record["controlled_bx_gate_label"]
                        not in row["controlled_bx_gates"]
                    ):
                        row["controlled_bx_gates"].append(
                            record["controlled_bx_gate_label"]
                        )
            if record.get("dependency_bx_gate_label") is not None:
                if record["dependency_bx_gate_state"] == "closed":
                    if (
                        record["dependency_bx_gate_label"]
                        not in row["dependency_bx_gates"]
                    ):
                        row["dependency_bx_gates"].append(
                            record["dependency_bx_gate_label"]
                        )

        html_lines.append("<tbody>")

        for _, row in rows.items():
            html_lines.append("<tr>")
            html_lines.append("<td>" + row["task"] + "</td>")
            html_lines.append("<td>" + row["task_state"] + "</td>")
            html_lines.append(
                "<td>" + "<br>".join(row["controlled_bx_gates"]) + "</td>"
            )
            html_lines.append(
                "<td>" + "<br>".join(row["dependency_bx_gates"]) + "</td>"
            )
            html_lines.append("<td>" + row["execution_outputs"] + "</td>")
            html_lines.append("</tr>")

        html_lines.append("</tbody>")

        html_lines.append("</table>")

        return "\n".join(html_lines)

    # ----------------------------------------------------------------------------------------
    def compose_bx_jobs_bx_tasks_bx_gates(self, records):
        """"""
        table = self.__prettyhelper.compose_bx_jobs_bx_tasks_bx_gates(records)

        return table.get_html_string()

    # ----------------------------------------------------------------------------------------
    def __render_execution_outputs(self, bx_task_directory, error_lines):

        # Filestore manager.
        execution_outputs = bx_filestores_get_default().get_runtime_execution_outputs(
            bx_task_directory
        )

        html_lines = []
        html_lines.append("<table class='T_execution_outputs'>")
        for execution_output in execution_outputs:
            # Exclude certain boring files.
            if execution_output.basename.startswith("bxflow"):
                continue
            if execution_output.bytes == 0:
                continue
            html_lines.append("<tr>")
            html_lines.append("<td class='T_bytes'>")
            html_lines.append(str(execution_output.bytes))
            html_lines.append("</td>")
            html_lines.append("<td class='T_filename'>")
            html_lines.append(self.__render_execution_output(execution_output))
            html_lines.append("</td>")
            html_lines.append("</tr>")
        html_lines.append("</table>")

        if error_lines is not None and error_lines != "":
            error_lines = error_lines.split("\n")
            html_lines.append("<hr>")
            html_lines.append("<div class='T_error_lines'>")
            for error_line in error_lines:
                html_lines.append(f"  <div class='T_error_line'>{error_line}</div")
            html_lines.append("</div>")

        return "\n".join(html_lines)

    # ----------------------------------------------------------------------------------------
    def __render_execution_output(self, execution_output):
        return '<a href="/filestore%s" target="_blank">%s</a>' % (
            execution_output.filename,
            execution_output.basename,
        )

    # ----------------------------------------------------------------------------------------
    def compose_bx_news(self, records):
        """"""
        table = self.__prettyhelper.compose_bx_news(records)

        return table.get_html_string()

    # ----------------------------------------------------------------------------------------
    def compose_bx_jobs(self, records):
        """"""

        field_names = [
            "created",
            "workflow",
            {"text": "dataset", "class": "T_dataset_column"},
            "state",
            {"text": "&nbsp;", "class": "T_details_column"},
            "&nbsp;",
            "rating",
            "comment",
        ]

        html_lines = []

        html_lines.append("<table>")
        html_lines.append("<thead>")
        html_lines.append("<tr>")
        for field_name in field_names:
            if isinstance(field_name, dict):
                html_lines.append(
                    f"<th class='{field_name['class']}'>{field_name['text']}</th>"
                )
            else:
                html_lines.append(f"<th>{field_name}</th>")
        html_lines.append("</tr>")
        html_lines.append("</thead>")

        html_lines.append("<tbody>")

        for record in records:
            bx_job_uuid = record["uuid"]
            html_lines.append(f"<tr bx_job_uuid='{bx_job_uuid}'>")
            html_lines.append(
                "<td class='T_created_column'>" + record["created_on"] + "</td>"
            )
            html_lines.append("<td>" + record["label"] + "</td>")
            html_lines.append(
                "<td class='T_dataset_column'>" + str(record["data_label"]) + "</td>"
            )
            state = str(record["state"])
            if state == BxJobStates.READY:
                state_label = "in progress"
            else:
                state_label = state
            html_lines.append(
                f"<td class='T_job_state T_{state}'><div>{state_label}</div></td>"
            )
            html_lines.append(
                f"<td bx_job_uuid='{bx_job_uuid}' class='T_bx_job_detail_job T_details_column'>details</td>"
            )
            html_lines.append("<td class='T_bx_job_status'>")
            if record["state"] == BxJobStates.READY:
                html_lines.append(
                    f"<div bx_job_uuid='{bx_job_uuid}' class='T_bx_job_cancel_job'>cancel</div>"
                )

            if record["state"] in [
                BxJobStates.SUCCEEDED,
                BxJobStates.FAILED,
                BxJobStates.CANCELLED,
                "blocked",
            ]:
                html_lines.append(
                    f"<div bx_job_uuid='{bx_job_uuid}' class='T_bx_job_delete_job'>delete</div>"
                )
            if record["state"] in [
                BxJobStates.FAILED,
            ]:
                html_lines.append(
                    f"<div bx_job_uuid='{bx_job_uuid}' class='T_bx_job_unblock_job'>retry</div>"
                )
            html_lines.append("</td>")

            # Rating column.
            rating_lines = []
            n_ratings = 5
            for i in range(1, n_ratings + 1):
                if i == record["rating"]:
                    css_class = "T_selected"
                else:
                    css_class = ""
                rating_lines.append(f"<div class='{css_class}' rating='{i}'>{i}</div>")
            html_lines.append(
                f"<td><div class='T_job_ratings'>{''.join(rating_lines)}</div></td>"
            )

            # Comment column.
            comment = record[BxJobFieldnames.COMMENT]
            if comment is None:
                comment = ""
            html_lines.append(
                "<td class='T_rubber_column'><TEXTAREA class='T_job_comment' rows=1>"
                + html.escape(comment)
                + "</TEXTAREA></td>"
            )

            html_lines.append("</tr>")

        html_lines.append("</tbody>")

        html_lines.append("</table>")

        return "\n".join(html_lines)

    # ----------------------------------------------------------------------------------------
    def compose_bx_variables(self, records):
        """"""
        table = self.__prettyhelper.compose_bx_variables(records)

        return table.get_html_string()

    # ----------------------------------------------------------------------------------------
    def compose_health_reports(self, health_reports):
        """"""

        table = self.__prettyhelper.compose_health_reports(health_reports)

        # Format the prettytable into html.
        html_string = table.get_html_string()

        html_string = html_string.replace("<br>", "\n")

        # Untangle the escaping which prettytable did to the cell contents.
        return html.unescape(html_string)

    # ----------------------------------------------------------------------------------------
    def compose_bx_jobs_data_grid(
        self,
        records,
        prepend_job_labels=None,
        append_job_labels=None,
        specific_job_labels=None,
        exclude_job_labels=None,
    ):
        """"""

        # Compose a prettytable of the records,
        # using private method to format each cell.
        table = self.__prettyhelper.compose_bx_jobs_data_grid(
            records,
            self.__compose_bx_jobs_data_grid_cell,
            prepend_job_labels=prepend_job_labels,
            append_job_labels=append_job_labels,
            specific_job_labels=specific_job_labels,
            exclude_job_labels=exclude_job_labels,
        )

        # Format the prettytable into html.
        html_string = table.get_html_string()

        # Untangle the escaping which prettytable did to the cell contents.
        return html.unescape(html_string)

    # ----------------------------------------------------------------------------------------
    def __compose_bx_jobs_data_grid_cell(self, data_cell):
        """"""

        uuid = data_cell.get("uuid")

        html_lines = []
        if uuid is not None:
            state = data_cell["state"]
            uuid = data_cell["uuid"]
            action_class = "T_bx_job_detail_job"
            html_lines.append(
                f"<div class='T_cell' bx_job_uuid='{uuid}' class='{action_class}'>"
            )
        # No job has ever been run for this cel?
        else:
            state = "NEVER"
            # Provide the cell with the two pieces of information it needs
            # to open the input panel for creating a new workflow.
            workflow_filename_classname = data_cell["workflow_filename_classname"]
            data_label = data_cell["data_label"]
            job_label = data_cell["job_label"]
            action_class = "T_bx_workflow_launch"
            html_lines.append(
                f"<div class='T_cell'"
                f" workflow_filename_classname='{workflow_filename_classname}'"
                f" data_label='{data_label}'"
                f" job_label='{job_label}'"
                f" class='{action_class}'>"
            )

        html_lines.append(f"<div class='T_box T_clickable T_{state}'></div>")
        html_lines.append("</div>")

        composed = "".join(html_lines)

        return composed

    # ----------------------------------------------------------------------------------------
    def compose_lines(self, lines):
        """"""
        html_string = []

        html_string.append("<div class='T_bx_composer_lines'>")
        for line in lines:
            html_string.append(
                f"<div class='T_bx_composer_line'>{html.escape(line)}</div>"
            )
        html_string.append("</div><!-- T_bx_composer_lines -->")

        return "\n".join(html_string)

    # ----------------------------------------------------------------------------------------
    def compose_tree(self, contents):
        """
        Compose the contents into a tree of sub-branches.
        """
        self.__lines = []
        self._compose_tree_branch("", contents)

        return "\n".join(self.__lines)

    # ----------------------------------------------------------------------------------------
    def _compose_tree_branch(self, key, contents):
        """
        Compose an HTML div, recursive.
        """
        prefix = " " * self.__indent
        self.__lines.append(f"{prefix}<div class='T_section'>")
        self.__indent += 2
        prefix = " " * self.__indent

        self.__lines.append(f"{prefix}<div class='T_title'>{html.escape(key)}</div>")
        self.__lines.append(f"{prefix}<div class='T_body'>")

        for key, content in contents.items():
            if isinstance(content, dict):
                self.__indent += 2
                self._compose_tree_branch(key, content)
                self.__indent -= 2
            else:
                self._compose_tree_leaf(key, content)

        self.__lines.append(f"{prefix}</div><!-- T_body -->")

        self.__indent -= 2
        prefix = " " * self.__indent
        self.__lines.append(f"{prefix}</div><!-- T_section -->")

    # ----------------------------------------------------------------------------------------
    def _compose_tree_leaf(self, key, value):
        """
        Componse an HTML input field.
        """
        self.__indent += 2
        prefix1 = " " * self.__indent
        prefix2 = " " * (self.__indent + 2)
        self.__lines.append(f"{prefix1}<div class='T_item'>")
        self.__lines.append(f"{prefix2}<div class='T_prompt'>{html.escape(key)}</div>")
        if isinstance(value, list):
            value = self.compose_lines(value)
        else:
            value = html.escape(str(value))
        self.__lines.append(f"{prefix2}<div class='T_value'>{value}</div>")
        self.__lines.append(f"{prefix1}</div>")
        self.__indent -= 2
