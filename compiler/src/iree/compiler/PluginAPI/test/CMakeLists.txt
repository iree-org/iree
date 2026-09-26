iree_add_all_subdirs()

# Dynamic loading is unsupported on Windows.
if(NOT IREE_BUILD_TESTS OR WIN32)
  return()
endif()

# Built here so the tests need no compiler of their own. Plain C and not
# renamed: their callbacks do not reference compiler C++ symbols.
foreach(_PLUGIN "valid_plugin" "invalid_plugin" "version_mismatch_plugin"
                "undefined_symbol_plugin" "abi_hash_mismatch_plugin"
                "null_field_plugin" "duplicate_id_plugin"
                "registration_failure_plugin" "static_collision_plugin")
  add_library(iree_compiler_PluginAPI_test_${_PLUGIN} MODULE
    "plugins/${_PLUGIN}.c")
  set_target_properties(iree_compiler_PluginAPI_test_${_PLUGIN} PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
endforeach()

foreach(_PLUGIN "valid_plugin" "abi_hash_mismatch_plugin" "undefined_symbol_plugin"
                "null_field_plugin" "duplicate_id_plugin"
                "registration_failure_plugin" "static_collision_plugin")
  target_link_libraries(iree_compiler_PluginAPI_test_${_PLUGIN}
    PRIVATE iree::compiler::PluginAPI::PluginEntryPoint)
endforeach()

iree_cc_test(
  NAME embedding_registration_test
  SRCS "embedding_registration_test.c"
  ARGS
    "$<TARGET_FILE:iree::compiler::API::SharedImpl>"
    "$<TARGET_FILE:iree_compiler_PluginAPI_test_registration_failure_plugin>"
  DEPS iree::compiler::bindings::c::loader
)
add_dependencies(iree_compiler_PluginAPI_test_embedding_registration_test
  iree::compiler::API::SharedImpl
  iree_compiler_PluginAPI_test_registration_failure_plugin
  iree_compiler_PluginAPI_test_static_collision_plugin)
target_link_libraries(iree_compiler_PluginAPI_test_static_collision_plugin PRIVATE
  iree::compiler::PluginAPI::Config::StaticLinkedPlugins)
add_test(NAME "iree/compiler/PluginAPI/test/embedding_static_collision"
  COMMAND iree_compiler_PluginAPI_test_embedding_registration_test
    "$<TARGET_FILE:iree::compiler::API::SharedImpl>"
    "$<TARGET_FILE:iree_compiler_PluginAPI_test_static_collision_plugin>")

foreach(_tool iree-compile iree-opt)
  add_test(NAME "iree/compiler/PluginAPI/test/${_tool}_registration_failure"
    COMMAND $<TARGET_FILE:${_tool}>
      "--iree-load-plugin=$<TARGET_FILE:iree_compiler_PluginAPI_test_registration_failure_plugin>"
      --help)
  set_tests_properties("iree/compiler/PluginAPI/test/${_tool}_registration_failure"
    PROPERTIES WILL_FAIL TRUE)
endforeach()

# ELF leaves undefined symbols alone; ld64 has to be told.
if(APPLE)
  target_link_options(iree_compiler_PluginAPI_test_undefined_symbol_plugin
    PRIVATE "-Wl,-undefined,dynamic_lookup")
endif()

# Out-of-tree plugins are normally built with hidden visibility; the macro must
# export the entry point itself.
add_library(iree_compiler_PluginAPI_test_hidden_visibility_plugin MODULE
  "plugins/hidden_visibility_plugin.cpp")
set_target_properties(iree_compiler_PluginAPI_test_hidden_visibility_plugin
  PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(iree_compiler_PluginAPI_test_hidden_visibility_plugin
  PRIVATE "-fvisibility=hidden")
target_link_libraries(iree_compiler_PluginAPI_test_hidden_visibility_plugin
  PRIVATE iree::compiler::PluginAPI::PluginEntryPoint)


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
    "plugin_with_deps.txt"
    "dynamic_plugin.txt"
    "abi_hash_mismatch.txt"
    "null_field.txt"
    "duplicate_id.txt"
  TOOLS
    FileCheck
    iree-compile
    iree-opt
)

iree_package_ns(_PACKAGE_NS)
string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
foreach(_TEST_SRC "invalid_plugin.txt" "valid_plugin.txt" "version_mismatch.txt"
                  "hidden_visibility.txt" "undefined_symbol.txt"
                  "plugin_with_deps.txt" "dynamic_plugin.txt"
                  "abi_hash_mismatch.txt" "null_field.txt" "duplicate_id.txt")
  set_property(TEST "${_PACKAGE_PATH}/${_TEST_SRC}.test" APPEND PROPERTY
    ENVIRONMENT
      "IREE_TEST_VALID_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_valid_plugin>"
      "IREE_TEST_INVALID_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_invalid_plugin>"
      "IREE_TEST_VERSION_MISMATCH_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_version_mismatch_plugin>"
      "IREE_TEST_HIDDEN_VISIBILITY_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_hidden_visibility_plugin>"
      "IREE_TEST_UNDEFINED_SYMBOL_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_undefined_symbol_plugin>"
      "IREE_TEST_ABI_HASH_MISMATCH_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_abi_hash_mismatch_plugin>"
      "IREE_TEST_NULL_FIELD_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_null_field_plugin>"
      "IREE_TEST_DUPLICATE_ID_PLUGIN=$<TARGET_FILE:iree_compiler_PluginAPI_test_duplicate_id_plugin>"
      $<$<BOOL:${IREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS}>:IREE_TEST_DEPS_PLUGIN=$<TARGET_FILE:iree_compiler_plugin_deps_plugin>>
      $<$<BOOL:${IREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS}>:IREE_TEST_DYNAMIC_PLUGIN=$<TARGET_FILE:iree_compiler_plugin_dynamic_test_plugin>>
  )
endforeach()

# A plugin naming MLIR symbols loads only if both sides renamed alike.
if(IREE_EXPERIMENTAL_COMPILER_DYNAMIC_PLUGINS)
  # Exercise a plugin with a separate helper library. Neither library is in
  # the compiler, so the plugin link must include both.
  iree_cc_library(
    NAME
      deps_helper
    HDRS
      "plugins/deps_helper.h"
    SRCS
      "plugins/deps_helper.cpp"
    PUBLIC
  )

  iree_cc_library(
    NAME
      deps_plugin
    SRCS
      "plugins/deps_plugin.cpp"
    DEPS
      ::deps_helper
      iree::compiler::PluginAPI
    PUBLIC
  )

  iree_compiler_register_dynamic_plugin(
    PLUGIN_ID
      deps_plugin
    TARGET
      ::deps_plugin
  )

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

endif()
