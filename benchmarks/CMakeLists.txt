iree_add_all_subdirs()

if (IREE_BUILD_EXPERIMENTAL_PYTHON_GENERATED_BENCHMARKS)
  include(generated_benchmark_suites.cmake)
endif()
