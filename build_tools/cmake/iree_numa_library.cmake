set(NUMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/numactl")
execute_process(COMMAND ./autogen.sh
  WORKING_DIRECTORY ${NUMA_SOURCE_DIR}
)
execute_process(COMMAND ./configure
  WORKING_DIRECTORY ${NUMA_SOURCE_DIR}
)