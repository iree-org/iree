iree_bytecode_module(
  NAME
    "$__TARGET_NAME"
  MODULE_FILE_NAME
    "$__MODULE_OUTPUT_PATH"
  SRC
    "$__SOURCE_MODEL_PATH"
  FLAGS
    $__COMPILE_FLAGS
  DEPENDS
    "$${_PACKAGE_NAME}_$__SOURCE_MODEL_TARGET"
)
# Mark dependency so that we have one target to drive them all.
add_dependencies(iree-benchmark-suites "$${_PACKAGE_NAME}_$__TARGET_NAME")
