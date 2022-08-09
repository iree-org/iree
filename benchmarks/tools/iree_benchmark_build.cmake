# If the source is a TFLite file, import it.
if("${_MODULE_SOURCE}" MATCHES "\.tflite$")
  cmake_path(GET _MODULE_SOURCE FILENAME _MODEL_BASENAME)
  set(_MODULE_SOURCE_TARGET "${_PACKAGE_NAME}_iree-import-tf-${_MODEL_BASENAME}")
  iree_import_tflite_model(
    TARGET_NAME "${_MODULE_SOURCE_TARGET}"
    SOURCE "${_MODULE_SOURCE}"
    OUTPUT_MLIR_FILE "${_MODULE_SOURCE}.mlir"
  )
  set(_MODULE_SOURCE "${_MODULE_SOURCE}.mlir")
endif()

# If the source is a TensorFlow SavedModel directory, import it.
if("${_MODULE_SOURCE}" MATCHES "-tf-model$")
  cmake_path(GET _MODULE_SOURCE FILENAME _MODEL_BASENAME)
  set(_MODULE_SOURCE_TARGET "${_PACKAGE_NAME}_iree-import-tf-${_MODEL_BASENAME}")
  iree_import_tf_model(
    TARGET_NAME "${_MODULE_SOURCE_TARGET}"
    SOURCE "${_MODULE_SOURCE}"
    ENTRY_FUNCTION "${_MODULE_ENTRY_FUNCTION}"
    OUTPUT_MLIR_FILE "${_MODULE_SOURCE}.mlir"
  )
  set(_MODULE_SOURCE "${_MODULE_SOURCE}.mlir")
endif()

iree_bytecode_module(
  NAME
  "${_COMPILATION_NAME}"
  MODULE_FILE_NAME
  "${_VMFB_FILE}"
  SRC
  "${_MODULE_SOURCE}"
  FLAGS
  ${_COMPILATION_ARGS}
  DEPENDS
  "${_MODULE_SOURCE_TARGET}"
  FRIENDLY_NAME
  "${_FRIENDLY_TARGET_NAME}"
)
