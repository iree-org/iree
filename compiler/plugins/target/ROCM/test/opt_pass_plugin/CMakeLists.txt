# Disabled on Windows until the shared library handling is cross-platform.
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  return()
endif()

iree_cc_library(
  NAME
    GPUHello
  SRCS
    "GPUHello.cpp"
  DEPS
    iree::compiler::API::Impl
  SHARED
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(iree_compiler_plugins_target_ROCM_test_opt_pass_plugin_GPUHello
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX "lib"
    OUTPUT_NAME "GPUHello"
)

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "gpu_hello.mlir"
  TOOLS
    FileCheck
    iree-opt
)
