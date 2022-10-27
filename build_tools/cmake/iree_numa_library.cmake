include(ExternalProject)

set(NUMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/numactl")
ExternalProject_Add(
  numa
  PREFIX ${NUMA_SOURCE_DIR}
  STAMP_DIR ${NUMA_SOURCE_DIR}
  SOURCE_DIR ${NUMA_SOURCE_DIR}
  CONFIGURE_COMMAND cd ${NUMA_SOURCE_DIR} && ${NUMA_SOURCE_DIR}/autogen.sh
  COMMAND ${NUMA_SOURCE_DIR}/configure
  INSTALL_COMMAND cmake -E echo "Skipping install step."
  BUILD_IN_SOURCE 1
)