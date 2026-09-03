iree_add_all_subdirs()

# The plugin mechanism is not yet supported on Windows.
if (NOT WIN32)

# Built rather than compiled inside a RUN line, so the tests need no compiler of
# their own. Plain C, and deliberately not renamed: they are never registered.
foreach(_PLUGIN "valid_plugin" "invalid_plugin" "version_mismatch_plugin")
  add_library(iree_compiler_PluginAPI_test_${_PLUGIN} MODULE
    "plugins/${_PLUGIN}.c")
  set_target_properties(iree_compiler_PluginAPI_test_${_PLUGIN} PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
endforeach()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "invalid_options.txt"
    "invalid_plugin.txt"
    "valid_plugin.txt"
    "version_mismatch.txt"
  TOOLS
    FileCheck
    iree-compile
    iree-opt
)

iree_package_ns(_PACKAGE_NS)
string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
foreach(_TEST_SRC "invalid_plugin.txt" "valid_plugin.txt" "version_mismatch.txt")
  set_property(TEST "${_PACKAGE_PATH}/${_TEST_SRC}.test" APPEND PROPERTY
    ENVIRONMENT
      "IREE_TEST_VALID_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_valid_plugin>"
      "IREE_TEST_INVALID_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_invalid_plugin>"
      "IREE_TEST_VERSION_MISMATCH_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_version_mismatch_plugin>"
  )
endforeach()

# A plugin naming MLIR symbols loads only if both sides renamed alike.
if(IREE_COMPILER_DYNAMIC_PLUGINS)
  iree_cc_library(
    NAME
      dynamic_plugin
    SRCS
      "dynamic_plugin.cpp"
    DEPS
      MLIRIR
    PUBLIC
  )

  iree_compiler_register_dynamic_plugin(
    PLUGIN_ID
      dynamic_test_plugin
    TARGET
      ::dynamic_plugin
  )

  add_test(
    NAME iree/compiler/PluginAPI/test/dynamic_plugin_load
    COMMAND
      iree-opt
      "--iree-load-plugin=$<TARGET_FILE:iree_compiler_plugin_dynamic_test_plugin>"
      --help
  )
  set_tests_properties(iree/compiler/PluginAPI/test/dynamic_plugin_load
    PROPERTIES
      PASS_REGULAR_EXPRESSION "DYNAMIC_TEST_PLUGIN: renamed MLIRContext ok"
  )
endif()

endif()
