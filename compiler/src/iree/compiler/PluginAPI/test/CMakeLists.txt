iree_add_all_subdirs()

# The plugin mechanism is not yet supported on Windows.
if (NOT WIN32)

# Built rather than compiled inside a RUN line, so the tests need no compiler of
# their own. Plain C, and deliberately not renamed: they are never registered.
if(IREE_BUILD_TESTS)
foreach(_PLUGIN "valid_plugin" "invalid_plugin" "version_mismatch_plugin"
                "undefined_symbol_plugin" "abi_hash_mismatch_plugin")
  add_library(iree_compiler_PluginAPI_test_${_PLUGIN} MODULE
    "plugins/${_PLUGIN}.c")
  set_target_properties(iree_compiler_PluginAPI_test_${_PLUGIN} PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
endforeach()

# valid_plugin reports the current ABI, so it takes the hash from the compiler.
target_include_directories(iree_compiler_PluginAPI_test_valid_plugin
  PRIVATE "${IREE_BINARY_DIR}/compiler/src")
add_dependencies(iree_compiler_PluginAPI_test_valid_plugin
  PluginABIHash_h__generate)

# Hidden visibility is what an out-of-tree plugin is normally built with, so
# the macro must export the entry point without help.
add_library(iree_compiler_PluginAPI_test_hidden_visibility_plugin MODULE
  "plugins/hidden_visibility_plugin.cpp")
set_target_properties(iree_compiler_PluginAPI_test_hidden_visibility_plugin
  PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(iree_compiler_PluginAPI_test_hidden_visibility_plugin
  PRIVATE "-fvisibility=hidden")
target_link_libraries(iree_compiler_PluginAPI_test_hidden_visibility_plugin
  PRIVATE iree::compiler::PluginAPI)


iree_lit_test_suite(
  NAME
    lit
  SRCS
    "invalid_options.txt"
    "invalid_plugin.txt"
    "valid_plugin.txt"
    "version_mismatch.txt"
    "hidden_visibility.txt"
    "undefined_symbol.txt"
    "abi_hash_mismatch.txt"
  TOOLS
    FileCheck
    iree-compile
    iree-opt
)

iree_package_ns(_PACKAGE_NS)
string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
foreach(_TEST_SRC "invalid_plugin.txt" "valid_plugin.txt" "version_mismatch.txt"
                  "hidden_visibility.txt" "undefined_symbol.txt"
                  "abi_hash_mismatch.txt")
  set_property(TEST "${_PACKAGE_PATH}/${_TEST_SRC}.test" APPEND PROPERTY
    ENVIRONMENT
      "IREE_TEST_VALID_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_valid_plugin>"
      "IREE_TEST_INVALID_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_invalid_plugin>"
      "IREE_TEST_VERSION_MISMATCH_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_version_mismatch_plugin>"
      "IREE_TEST_HIDDEN_VISIBILITY_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_hidden_visibility_plugin>"
      "IREE_TEST_UNDEFINED_SYMBOL_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_undefined_symbol_plugin>"
      "IREE_TEST_ABI_HASH_MISMATCH_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_abi_hash_mismatch_plugin>"
  )
endforeach()
endif()


endif()
