iree_bytecode_module(
  NAME
    "$_TARGET_NAME_"
  MODULE_FILE_NAME
    "$_MODULE_OUTPUT_PATH_"
  SRC
    "$_SOURCE_MODEL_PATH_"
  FLAGS
    $_COMPILE_FLAGS_
  DEPENDS
    "$${_PACKAGE_NAME}_$_SOURCE_MODEL_TARGET_"
)
# Mark dependency so that we have one target to drive them all.
add_dependencies(iree-benchmark-suites "$${_PACKAGE_NAME}_$_TARGET_NAME_")
