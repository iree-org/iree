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
    "version_mismatch.c"
  TOOLS
    FileCheck
    iree-compile
    iree-opt
)

# These build a plugin, so give lit the compiler this build used rather than
# leaving %host_cc to whatever PATH offers.
iree_package_ns(_PACKAGE_NS)
string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
foreach(_TEST_SRC "invalid_plugin.c" "valid_plugin.c" "version_mismatch.c")
  set_property(TEST "${_PACKAGE_PATH}/${_TEST_SRC}.test"
    APPEND PROPERTY ENVIRONMENT "IREE_HOST_CC=${CMAKE_C_COMPILER}"
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
