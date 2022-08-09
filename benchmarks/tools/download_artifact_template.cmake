add_custom_command(
  OUTPUT "$_ARTIFACT_LOCAL_PATH_"
  COMMAND
    "$${Python3_EXECUTABLE}" "$${IREE_ROOT_DIR}/build_tools/scripts/download_file.py"
    "$_ARTIFACT_SOURCE_URL_" -o "$_ARTIFACT_LOCAL_PATH_"
  DEPENDS
    "$${IREE_ROOT_DIR}/build_tools/scripts/download_file.py"
  COMMENT "Downloading $_ARTIFACT_SOURCE_URL_"
)
