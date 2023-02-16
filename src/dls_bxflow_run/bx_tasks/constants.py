class Keywords:
    PREPARE_ENVIRONMENT = "prepare_environment"
    NEEDS_DATAFACE = "needs_dataface"


class ExtractionErrorLinesMessages:
    PROBLEM_READING = "problem reading"
    EXISTS_BUT_IS_EMPTY = "exists but is empty"
    DOES_NOT_EXIST = "does not exist"


class Types:
    WHATENV = "dls_bxflow_run.bx_tasks.whatenv"
    DUMMY = "dls_bxflow_run.bx_tasks.dummy"
    SYMLINK = "dls_bxflow_run.bx_tasks.symlink"
    JUPYTER = "dls_bxflow_run.bx_tasks.jupyter"
    FILENAME_CLASSNAME = "dls_bxflow_run.bx_tasks.filename_classname"
    MODULE_CLASSNAME = "dls_bxflow_run.bx_tasks.module_classname"
    PTYPY_MPI = "dls_bxflow_run.bx_tasks.ptypy_mpi"
    PTYREX_MPI = "dls_bxflow_run.bx_tasks.ptyrex_mpi"
