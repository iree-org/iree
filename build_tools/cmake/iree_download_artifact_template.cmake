add_custom_command(
  OUTPUT "$_OUTPUT_PATH_"
  COMMAND
    "$${Python3_EXECUTABLE}" "$${IREE_ROOT_DIR}/build_tools/scripts/download_file.py"
    "$_SOURCE_URL_" -o "$_OUTPUT_PATH_"
  DEPENDS
    "$${IREE_ROOT_DIR}/build_tools/scripts/download_file.py"
  COMMENT "Downloading $_SOURCE_URL_"
)
add_custom_target(
    "$${_PACKAGE_NAME}_$_TARGET_NAME_"
  DEPENDS
    "$_OUTPUT_PATH_"
)
