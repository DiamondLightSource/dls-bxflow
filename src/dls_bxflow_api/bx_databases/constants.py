# ----------------------------------------------------------------------------------------
class NewsFieldnames:
    UUID = "uuid"
    CREATED_ON = "created_on"
    BX_JOB_UUID = "bx_job_uuid"
    BX_TASK_UUID = "bx_task_uuid"
    TOPIC = "topic"
    HEADLINE = "headline"
    DETAILS = "details"


# ----------------------------------------------------------------------------------------
class BxJobFieldnames:
    UUID = "uuid"
    CREATED_ON = "created_on"
    TYPE = "type"
    STATE = "state"
    SPECIFICATION = "specification"
    LABEL = "label"
    DATA_LABEL = "data_label"
    DIRECTORY = "directory"
    BEAMLINE = "beamline"
    VISIT = "visit"
    BX_CATALOG_UUID = "bx_catalog_uuid"
    BX_WORKFLOW_UUID = "bx_workflow_uuid"
    COMMENT = "comment"
    RATING = "rating"


# ----------------------------------------------------------------------------------------
class BxGateFieldnames:
    UUID = "uuid"
    CREATED_ON = "created_on"
    TYPE = "type"
    BX_JOB_UUID = "bx_job_uuid"
    BX_TASK_UUID = "bx_task_uuid"
    STATE = "state"
    SPECIFICATION = "specification"
    LABEL = "label"


# ----------------------------------------------------------------------------------------
class BxLauncherFieldnames:
    UUID = "uuid"
    CREATED_ON = "created_on"
    TYPE = "type"
    STATE = "state"
    SPECIFICATION = "specification"
    SUBMIT_COUNT = "submit_count"

    # Name of the cluster this launcher uses to launch jobs on.
    # See dls_bxflow_api.remex for enumerated values.
    REMEX_CLUSTER = "remex_cluster"


# ----------------------------------------------------------------------------------------
class BxTaskFieldnames:
    UUID = "uuid"
    CREATED_ON = "created_on"
    TYPE = "type"
    BX_JOB_UUID = "bx_job_uuid"
    BX_LAUNCHER_UUID = "bx_launcher_uuid"
    launch_info = "launch_info"
    STATE = "state"
    SPECIFICATION = "specification"
    LABEL = "label"
    DIRECTORY = "directory"
    EXIT_CODE = "exit_code"
    ERROR_LINES = "error_lines"


# ----------------------------------------------------------------------------------------
class RelationFieldnames:
    LHS = "lhs"
    RHS = "rhs"


# ----------------------------------------------------------------------------------------
class BxVariableFieldnames:
    UUID = "uuid"
    CREATED_ON = "created_on"
    TYPE = "type"
    BX_JOB_UUID = "bx_job_uuid"
    STATE = "state"
    NAME = "name"
    VALUE = "value"


# ----------------------------------------------------------------------------------------
class BxWorkflowFieldnames:
    UUID = "uuid"
    CREATED_ON = "created_on"
    BX_JOB_UUID = "bx_job_uuid"
    FILENAME_CLASSNAME = "filename_classname"
    BX_SETTINGS_JSON = "bx_settings_json"


# ----------------------------------------------------------------------------------------
class RevisionFieldnames:
    CREATED_ON = "created_on"
    NUMBER = "number"
