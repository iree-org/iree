add_custom_command(
  OUTPUT "$_OUTPUT_PATH_"
  COMMAND
    "$${Python3_EXECUTABLE}" "$${IREE_ROOT_DIR}/build_tools/scripts/generate_flagfile.py"
      --module_file="$_REL_MODULE_FILE_PATH_"
      --device=$_DRIVER_
      --entry_function=$_ENTRY_FUNCTION_
      --function_inputs=$_FUNCTION_INPUTS_
      --additional_args="$_EXTRA_FLAGS_"
      -o "$_OUTPUT_PATH_"
  DEPENDS
    "$${IREE_ROOT_DIR}/build_tools/scripts/generate_flagfile.py"
    "$_MODULE_FILE_PATH_"
  COMMENT "Generating $_OUTPUT_PATH_"
)
add_custom_target(
    "$${_PACKAGE_NAME}_$_TARGET_NAME_"
  DEPENDS
    "$_OUTPUT_PATH_"
)
# Mark dependency so that we have one target to drive them all.
add_dependencies(iree-benchmark-suites "$${_PACKAGE_NAME}_$_TARGET_NAME_")
