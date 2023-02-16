# Remex (remote execution) API.
# TODO: Reconsider package/module placement for remex keywords.

from dls_bxflow_api.exceptions import NotFound

# The following keywords are honored:
# cluster
# redhat_release
# m_mem_free
# h_rt
# q
# pe
# gpu
# gpu_arch
# h


class Keywords:
    HINTS = "remex_hints"
    CLUSTER = "cluster"
    REDHAT_RELEASE = "redhat_release"
    MEMORY_LIMIT = "m_mem_free"
    TIME_LIMIT = "h_rt"
    QUEUE = "q"
    PARALLEL_ENVIRONMENT = "pe"
    GPU = "gpu"
    GPU_ARCH = "gpu_arch"
    HOST = "host"


class Clusters:
    HAMILTON = "dls_bxflow_api::remex::cluster::hamilton"
    SCIENCE = "dls_bxflow_api::remex::cluster::global/cluster"
    TEST = "dls_bxflow_api::remex::cluster::global/testcluster"
    LOCAL = "dls_bxflow_api::remex::cluster::local"

    __list = [HAMILTON, SCIENCE, TEST, LOCAL]

    def validate(name):
        if name not in Clusters.__list:
            raise NotFound(
                'cluster name "%s" is not in %s' % (name, str(Clusters.__list))
            )
