import logging

import prettytable

# Base class for generic things.
from dls_bxflow_api.thing import Thing

# Class to do the work using prettytable.
from dls_bxflow_lib.bx_composers.prettyhelper import PrettyHelper

# Filestore manager.
from dls_bxflow_lib.bx_filestores.bx_filestores import bx_filestores_get_default

logger = logging.getLogger(__name__)

thing_type = "dls_bxflow_lib.bx_composers.text"


class Text(Thing):
    """ """

    # ----------------------------------------------------------------------------------------
    def __init__(self, specification=None):
        Thing.__init__(self, thing_type, specification)

        self.__prettyhelper = PrettyHelper()

    # ----------------------------------------------------------------------------------------
    def compose_bx_jobs_bx_tasks_bx_gates(self, records):
        """"""
        table = prettytable.PrettyTable()
        table.field_names = [
            "job",
            "task",
            "task state",
            "controlled gates",
            "dependency gates",
        ]

        rows = {}
        for record in records:
            if rows.get(record["bx_task_uuid"]) is None:
                rows[record["bx_task_uuid"]] = {
                    "job": record["bx_job_label"],
                    "task": record["bx_task_label"],
                    "task_state": record["bx_task_state"],
                    "controlled_bx_gates": [],
                    "dependency_bx_gates": [],
                }
            row = rows[record["bx_task_uuid"]]
            if record.get("controlled_bx_gate_label") is not None:
                bx_gate_state = "%s: %s" % (
                    record["controlled_bx_gate_label"],
                    record["controlled_bx_gate_state"],
                )
                if bx_gate_state not in row["controlled_bx_gates"]:
                    row["controlled_bx_gates"].append(bx_gate_state)
            if record.get("dependency_bx_gate_label") is not None:
                bx_gate_state = "%s: %s" % (
                    record["dependency_bx_gate_label"],
                    record["dependency_bx_gate_state"],
                )
                if bx_gate_state not in row["dependency_bx_gates"]:
                    row["dependency_bx_gates"].append(bx_gate_state)

        for _, row in rows.items():
            row2 = []
            row2.append(row["job"])
            row2.append(row["task"])
            row2.append(row["task_state"])
            row2.append("\n".join(row["controlled_bx_gates"]))
            row2.append("\n".join(row["dependency_bx_gates"]))
            table.add_row(row2)

        table.align = "l"

        return table

    # ----------------------------------------------------------------------------------------
    def compose_bx_job_details(self, bx_job_record, bx_jobs_bx_tasks_bx_gates_records):
        """ """

        table = prettytable.PrettyTable()
        table.field_names = [
            "task",
            "task state",
            "exit code",
            "has opened",
            "waiting for",
            "execution outputs (with size)",
        ]

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
                    "task_exit_code": record["bx_task_exit_code"],
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

        for _, row in rows.items():
            row2 = []
            row2.append(row["task"])
            row2.append(row["task_state"])
            row2.append(row["task_exit_code"])
            row2.append("\n".join(row["controlled_bx_gates"]))
            row2.append("\n".join(row["dependency_bx_gates"]))
            row2.append(row["execution_outputs"])
            table.add_row(row2)

        table.align = "l"
        table.title = "Task Details (job state %s)" % (bx_job_record["state"])

        return table.get_string()

    # ----------------------------------------------------------------------------------------
    def __render_execution_outputs(self, bx_task_directory, error_lines):

        # Filestore manager.
        execution_outputs = bx_filestores_get_default().get_runtime_execution_outputs(
            bx_task_directory
        )

        lines = []

        for execution_output in execution_outputs:
            # Exclude certain boring files.
            if execution_output.basename.startswith("bxflow"):
                continue
            if execution_output.bytes == 0:
                continue

            line = "%8d | %s" % (execution_output.bytes, execution_output.filename)

            lines.append(line)

        if error_lines is not None and error_lines != "":
            error_lines = error_lines.split("\n")
            lines.append("-" * 32)
            for error_line in error_lines:
                lines.append(error_line)

        lines.append("")

        return "\n".join(lines)

    # ----------------------------------------------------------------------------------------
    def __render_execution_output(self, execution_output):
        return

    # ----------------------------------------------------------------------------------------
    def compose_bx_news(self, records):
        """"""
        table = self.__prettyhelper.compose_bx_news(records)

        return table.get_string()

    # ----------------------------------------------------------------------------------------
    def compose_bx_jobs(self, records):
        """"""
        table = self.__prettyhelper.compose_bx_jobs(records)

        return table.get_string()

    # ----------------------------------------------------------------------------------------
    def compose_bx_variables(self, records):
        """"""
        table = self.__prettyhelper.compose_bx_variables(records)

        return table.get_string()

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
        table = self.__prettyhelper.compose_bx_jobs_data_grid(
            records,
            self.__compose_bx_jobs_data_grid_cell,
            prepend_job_labels=prepend_job_labels,
            append_job_labels=append_job_labels,
            specific_job_labels=specific_job_labels,
            exclude_job_labels=exclude_job_labels,
        )

        return table.get_string()

    # ----------------------------------------------------------------------------------------
    def __compose_bx_jobs_data_grid_cell(self, data_cell):
        """"""

        uuid = data_cell.get("uuid")
        if uuid is not None:
            composed = data_cell["state"]
        else:
            composed = "-"

        return composed
