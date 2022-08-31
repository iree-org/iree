add_custom_command(
  OUTPUT "$__OUTPUT_PATH"
  COMMAND
    "$${Python3_EXECUTABLE}" "$${IREE_ROOT_DIR}/build_tools/scripts/generate_flagfile.py"
      --module_file="$__REL_MODULE_FILE_PATH"
      --device="$__DRIVER"
      --entry_function="$__ENTRY_FUNCTION"
      --function_inputs="$__FUNCTION_INPUTS"
      --additional_args="$__EXTRA_FLAGS"
      -o "$__OUTPUT_PATH"
  DEPENDS
    "$${IREE_ROOT_DIR}/build_tools/scripts/generate_flagfile.py"
    "$__MODULE_FILE_PATH"
  COMMENT "Generating $__OUTPUT_PATH"
)
add_custom_target(
    "$${_PACKAGE_NAME}_$__TARGET_NAME"
  DEPENDS
    "$__OUTPUT_PATH"
)
# Mark dependency so that we have one target to drive them all.
add_dependencies(iree-benchmark-suites "$${_PACKAGE_NAME}_$__TARGET_NAME")
