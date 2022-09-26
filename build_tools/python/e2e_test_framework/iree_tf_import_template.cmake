# Import the Tensorflow model "$__SOURCE_MODEL_PATH"
iree_import_tf_model(
  TARGET_NAME "$${_PACKAGE_NAME}_$__TARGET_NAME"
  SOURCE "$__SOURCE_MODEL_PATH"
  ENTRY_FUNCTION "$__ENTRY_FUNCTION"
  OUTPUT_MLIR_FILE "$__OUTPUT_PATH"
)
# Mark dependency so users can import models without compiling them.
add_dependencies(iree-benchmark-import-models "$${_PACKAGE_NAME}_$__TARGET_NAME")
