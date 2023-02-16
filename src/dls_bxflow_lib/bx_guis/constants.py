class Keywords:
    COMMAND = "bx_guis::keywords::command"
    PAYLOAD = "bx_guis::keywords::payload"
    TAB_ID = "bx_guis::keywords::tab_id"
    BX_JOB_UUID = "bx_job_uuid"
    JOB_COMMENT = "job_comment"
    JOB_RATING = "job_rating"


class Commands:
    LOAD_TABS = "bx_guis::commands::load_tabs"
    SELECT_TAB = "bx_guis::commands::select_tab"
    CANCEL_JOB = "bx_guis::commands::cancel_job"
    DELETE_JOB = "bx_guis::commands::delete_job"
    UNBLOCK_JOB = "bx_guis::commands::unblock_job"
    UPDATE_JOB = "bx_guis::commands::update_job"
    GET_RECENT_JOBS = "bx_guis::commands::get_recent_jobs"
    GET_RECENT_NEWS = "bx_guis::commands::get_recent_news"
    GET_JOB_DATA_GRID = "bx_guis::commands::get_job_data_grid"
    GET_SYSTEM_HEALTH = "bx_guis::commands::get_system_health"
    SHOW_WORKFLOW_SETTINGS = "bx_guis::commands::show_workflow_settings"
    START_WORKFLOW = "bx_guis::commands::start_workflow"
    START_WORKFLOW_NOCOOKIE = "bx_guis::commands::start_workflow_nocookie"
    GET_WORKFLOW_CONSTRUCTOR_KWARGS = (
        "bx_guis::commands::get_workflow_constructor_kwargs"
    )
    GET_JOB_NEWS = "bx_guis::commands::get_job_news"
    GET_JOB_DETAILS = "bx_guis::commands::get_job_details"
    GET_JOB_VARIABLES = "bx_guis::commands::get_job_variables"
    QUERY_LOGSTORE = "bx_guis::commands::query_logstore"


class Cookies:
    TABS_MANAGER = "BXFLOW_TABS_MANAGER"
    RECENT_JOBS_UX = "BXFLOW_RECENT_JOBS_UX"
    JOB_DETAILS_UX = "BXFLOW_JOB_DETAILS_UX"
    JOB_NEWS_UX = "BXFLOW_JOB_NEWS_UX"
    JOB_VARIABLES_UX = "BXFLOW_JOB_VARIABLES_UX"
    JOB_DATA_GRID_UX = "BXFLOW_JOB_DATA_GRID_UX"
    SYSTEM_HEALTH_UX = "BXFLOW_SYSTEM_HEALTH_UX"
    JOB_SUBMIT_UX = "BXFLOW_JOB_SUBMIT_UX"
    ERROR_LOGS_UX = "BXFLOW_ERROR_LOGS_UX"
