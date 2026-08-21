iree_add_all_subdirs()

# The plugin mechanism is not yet supported on Windows.
if (NOT WIN32)

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "invalid_options.txt"
    "invalid_plugin.c"
    "valid_plugin.c"
  TOOLS
    FileCheck
    iree-compile
    iree-opt
)

# End-to-end check of the renamed compiler ABI: a plugin referencing MLIR C++
# symbols only loads if libIREECompiler exports them under the renamed
# spelling and the plugin archive was renamed to match.
if(IREE_COMPILER_DYNAMIC_PLUGINS)
  include(iree_renamed_link)
  iree_compiler_dynamic_plugin(
    NAME
      iree_compiler_PluginAPI_test_dynamic_plugin
    SRCS
      dynamic_plugin.cpp
  )
  add_test(
    NAME iree/compiler/PluginAPI/test/dynamic_plugin_load
    COMMAND
      iree-opt
      "--iree-load-plugin=dynamic_test_plugin=$<TARGET_FILE:iree_compiler_PluginAPI_test_dynamic_plugin>"
      --help
  )
  set_tests_properties(iree/compiler/PluginAPI/test/dynamic_plugin_load
    PROPERTIES
      PASS_REGULAR_EXPRESSION "DYNAMIC_TEST_PLUGIN: renamed MLIRContext ok"
  )
endif()

endif()
