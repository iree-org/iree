iree_add_all_subdirs()

iree_compiler_register_plugin(
  PLUGIN_ID
    input_stablehlo
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
    MLIRTransforms
    iree::compiler::PluginAPI
    stablehlo-iree::InputConversion::InputConversion
  PUBLIC
)
