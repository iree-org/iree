iree_add_all_subdirs()

iree_compiler_register_plugin(
  PLUGIN_ID
    input_tosa
  TARGET
    ::registration
)

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    MLIRIR
    MLIRPass
    MLIRTosaDialect
    MLIRTosaToArith
    MLIRTosaToLinalg
    MLIRTosaToSCF
    MLIRTosaToTensor
    MLIRTransforms
    iree::compiler::PluginAPI
    tosa-iree::InputConversion::InputConversion
  PUBLIC
)
