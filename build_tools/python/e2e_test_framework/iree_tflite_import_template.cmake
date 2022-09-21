# Import the TFLite model "$__SOURCE_MODEL_PATH"
iree_import_tflite_model(
  TARGET_NAME "$${_PACKAGE_NAME}_$__TARGET_NAME"
  SOURCE "$__SOURCE_MODEL_PATH"
  OUTPUT_MLIR_FILE "$__OUTPUT_PATH"
)
