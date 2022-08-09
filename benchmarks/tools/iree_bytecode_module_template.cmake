iree_bytecode_module(
  NAME
    "$_MODULE_TARGET_NAME_"
  MODULE_FILE_NAME
    "$_OUTPUT_PATH_"
  SRC
    "$_MLIR_LOCAL_PATH_"
  FLAGS
    $_COMPILE_FLAGS_
  DEPENDS
    "$_SOURCE_MODEL_TARGET_"
)

set(_MODULE_TARGET "$${_PACKAGE_NAME}_$_MODULE_TARGET_NAME_")
add_dependencies(iree-benchmark-suites "$${_MODULE_TARGET}")
add_dependencies("$${SUITE_SUB_TARGET}" "$${_MODULE_TARGET}")
