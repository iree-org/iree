# Fetch the model from "$__SOURCE_URL"
add_custom_command(
  OUTPUT "$__OUTPUT_PATH"
  COMMAND
    "$${Python3_EXECUTABLE}" "$${IREE_ROOT_DIR}/build_tools/scripts/download_file.py"
    "$__SOURCE_URL" -o "$__OUTPUT_PATH"
  DEPENDS
    "$${IREE_ROOT_DIR}/build_tools/scripts/download_file.py"
  COMMENT "Downloading $__SOURCE_URL"
)
add_custom_target(
    "$${_PACKAGE_NAME}_$__TARGET_NAME"
  DEPENDS
    "$__OUTPUT_PATH"
)
