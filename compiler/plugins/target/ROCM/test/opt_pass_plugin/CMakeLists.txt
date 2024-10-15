add_library(GPUHello SHARED GPUHello.cpp)
target_link_libraries(GPUHello
  PRIVATE
  LLVMSupport
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(GPUHello
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX "lib"
    OUTPUT_NAME "GPUHello"
)

target_compile_options(GPUHello PRIVATE ${IREE_DEFAULT_COPTS})

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "gpu_hello.mlir"
  TOOLS
    FileCheck
    iree-opt
  LABELS
    "noasan"
)
