# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_wasm_cc_library()
#
# CMake function for declaring JS companion files that accompany a C library
# compiled for wasm. These are the CMake equivalent of the Bazel
# iree_wasm_cc_library rule.
#
# Creates an INTERFACE library target with custom properties:
#   IREE_WASM_JS_MODULE: the wasm import module name (e.g., "iree_syscall").
#   IREE_WASM_JS_SRCS: semicolon-separated list of absolute paths to JS files.
#   IREE_WASM_JS_ENTRIES: semicolon-separated list of "module:path" entries,
#       propagated transitively via genex chains through iree_cc_library deps.
#
# Parameters:
#   NAME: target name.
#   MODULE: wasm import module name.
#   SRCS: list of JS source files.
#   DEPS: list of other iree_wasm_cc_library targets.
#   PUBLIC: whether to export under iree::.
#   TESTONLY: whether this is test-only.
#
# Example:
#   iree_wasm_cc_library(
#     NAME
#       syscall_imports
#     SRCS
#       "src/syscall_imports.js"
#     MODULE
#       "iree_syscall"
#     PUBLIC
#   )

function(iree_wasm_cc_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "PACKAGE;NAME;MODULE"
    "SRCS;DEPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: iree_package_name.
  if(_RULE_PACKAGE)
    set(_PACKAGE_NS "${_RULE_PACKAGE}")
    string(REPLACE "::" "_" _PACKAGE_NAME ${_RULE_PACKAGE})
  else()
    iree_package_ns(_PACKAGE_NS)
    iree_package_name(_PACKAGE_NAME)
  endif()
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  # Replace dependencies passed by ::name with iree::package::name.
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Create an INTERFACE library. This target carries no C compilation info —
  # it exists only to participate in the dependency graph and carry JS file
  # metadata via custom properties.
  add_library(${_NAME} INTERFACE)

  # Convert source file paths to absolute.
  set(_ABS_SRCS)
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_ABS_SRC "${_SRC}" ABSOLUTE)
    list(APPEND _ABS_SRCS "${_ABS_SRC}")
  endforeach()

  # Build the module:path entries list for transitive propagation.
  set(_JS_ENTRIES "")
  foreach(_SRC ${_ABS_SRCS})
    list(APPEND _JS_ENTRIES "${_RULE_MODULE}:${_SRC}")
  endforeach()

  # Set custom properties for wasm JS companion metadata.
  set_target_properties(${_NAME} PROPERTIES
    IREE_WASM_JS_MODULE "${_RULE_MODULE}"
    IREE_WASM_JS_SRCS "${_ABS_SRCS}"
    IREE_WASM_JS_ENTRIES "${_JS_ENTRIES}"
  )

  # Link deps so they're in the transitive dependency graph.
  if(_RULE_DEPS)
    target_link_libraries(${_NAME} INTERFACE ${_RULE_DEPS})
  endif()

  # IDE folder organization.
  if(_RULE_PUBLIC)
    set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER})
  elseif(_RULE_TESTONLY)
    set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)
  else()
    set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/internal)
  endif()

  # Install target into the export set so that PUBLIC cc_libraries that depend
  # on this target can be exported without CMake complaining about missing
  # transitive dependencies.
  if(NOT _RULE_TESTONLY)
    iree_install_targets(
      TARGETS ${_NAME}
    )
  endif()

  # Alias the target to iree::package::name.
  iree_add_alias_library(${_PACKAGE_NS}::${_RULE_NAME} ${_NAME})

  if(NOT "${_PACKAGE_NS}" STREQUAL "")
    iree_package_dir(_PACKAGE_DIR)
    if("${_RULE_NAME}" STREQUAL "${_PACKAGE_DIR}")
      iree_add_alias_library(${_PACKAGE_NS} ${_NAME})
    endif()
  endif()
endfunction()

# iree_wasm_entry()
#
# CMake function for declaring the JS entry point for a wasm binary. These
# are the CMake equivalent of the Bazel iree_wasm_entry rule.
#
# Creates an INTERFACE library target with custom properties:
#   IREE_WASM_ENTRY_MAIN: absolute path to the entry point .mjs file.
#   IREE_WASM_ENTRY_SRCS: semicolon-separated list of absolute paths to
#       local imports of the entry point (bundled at build time).
#
# Entry point discovery is done at generation time via direct dep checking
# in iree_cc_test / iree_wasm_cc_binary — no transitive propagation needed
# because the entry target is always a direct dep ("just add :main to deps").
#
# Parameters:
#   NAME: target name.
#   MAIN: entry point JS file.
#   SRCS: local imports of the entry point.
#   PUBLIC: whether to export under iree::.
#
# Example:
#   iree_wasm_entry(
#     NAME
#       main
#     MAIN
#       "proactor_worker_main.mjs"
#     SRCS
#       "proactor_event_host.mjs"
#       "proactor_ring.mjs"
#     PUBLIC
#   )

function(iree_wasm_entry)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "PACKAGE;NAME;MAIN"
    "SRCS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name.
  if(_RULE_PACKAGE)
    set(_PACKAGE_NS "${_RULE_PACKAGE}")
    string(REPLACE "::" "_" _PACKAGE_NAME ${_RULE_PACKAGE})
  else()
    iree_package_ns(_PACKAGE_NS)
    iree_package_name(_PACKAGE_NAME)
  endif()
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  # Create an INTERFACE library for dependency graph participation.
  add_library(${_NAME} INTERFACE)

  # Resolve entry point and source paths to absolute.
  get_filename_component(_MAIN_ABS "${_RULE_MAIN}" ABSOLUTE)
  set(_ABS_SRCS)
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_ABS_SRC "${_SRC}" ABSOLUTE)
    list(APPEND _ABS_SRCS "${_ABS_SRC}")
  endforeach()

  # Set custom properties for entry point metadata.
  set_target_properties(${_NAME} PROPERTIES
    IREE_WASM_ENTRY_MAIN "${_MAIN_ABS}"
    IREE_WASM_ENTRY_SRCS "${_ABS_SRCS}"
  )

  # IDE folder organization.
  if(_RULE_PUBLIC)
    set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER})
  elseif(_RULE_TESTONLY)
    set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/test)
  else()
    set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/internal)
  endif()

  # Install target into the export set (same reason as iree_wasm_cc_library).
  if(NOT _RULE_TESTONLY)
    iree_install_targets(
      TARGETS ${_NAME}
    )
  endif()

  # Alias the target to iree::package::name.
  iree_add_alias_library(${_PACKAGE_NS}::${_RULE_NAME} ${_NAME})

  if(NOT "${_PACKAGE_NS}" STREQUAL "")
    iree_package_dir(_PACKAGE_DIR)
    if("${_RULE_NAME}" STREQUAL "${_PACKAGE_DIR}")
      iree_add_alias_library(${_PACKAGE_NS} ${_NAME})
    endif()
  endif()
endfunction()

# _iree_wasm_setup_bundler()
#
# Sets up the wasm binary bundler for a target. Used by both iree_wasm_cc_binary
# and iree_cc_test (wasm path).
#
# This function:
#   1. Propagates IREE_WASM_JS_ENTRIES and IREE_WASM_JS_SRCS from deps using
#      the same $<GENEX_EVAL:$<TARGET_PROPERTY:...>> pattern as
#      _iree_cc_library_add_object_deps. This makes the collection fully
#      order-independent — targets don't need to exist at configure time.
#   2. Discovers the entry point from direct deps via generation-time genexes.
#   3. Generates a manifest file via file(GENERATE) (evaluated at generation
#      time, after all CMakeLists are processed).
#   4. Creates an add_custom_command to run the bundler at build time.
#
# Parameters:
#   WASM_TARGET: name of the wasm executable target.
#   OUTPUT: path for the output .mjs file.
#   MAIN: explicit entry point (optional; discovered from deps if not set).
#   DEPS: list of dependencies to collect wasm metadata from.
function(_iree_wasm_setup_bundler)
  cmake_parse_arguments(
    _RULE
    ""
    "WASM_TARGET;OUTPUT;MAIN"
    "DEPS"
    ${ARGN}
  )

  set(_BUNDLER "${CMAKE_SOURCE_DIR}/build_tools/wasm/wasm_binary_bundler.py")
  set(_MANIFEST_FILE "${_RULE_OUTPUT}.modules")

  # Propagate wasm JS metadata from deps via genex chains.
  # At configure time, these are just recipe strings containing genex
  # expressions. At generation time, CMake resolves the full chain — the dep
  # targets don't need to exist when this code runs.
  foreach(_DEP ${_RULE_DEPS})
    set_property(TARGET ${_RULE_WASM_TARGET} APPEND PROPERTY
      IREE_WASM_JS_ENTRIES
      "$<GENEX_EVAL:$<TARGET_PROPERTY:${_DEP},IREE_WASM_JS_ENTRIES>>"
    )
    set_property(TARGET ${_RULE_WASM_TARGET} APPEND PROPERTY
      IREE_WASM_JS_SRCS
      "$<GENEX_EVAL:$<TARGET_PROPERTY:${_DEP},IREE_WASM_JS_SRCS>>"
    )
  endforeach()

  # Resolve JS companion entries and source files at generation time.
  set(_JS_ENTRIES_RAW "$<GENEX_EVAL:$<TARGET_PROPERTY:${_RULE_WASM_TARGET},IREE_WASM_JS_ENTRIES>>")
  set(_JS_ENTRIES "$<FILTER:$<REMOVE_DUPLICATES:${_JS_ENTRIES_RAW}>,INCLUDE,.+>")
  set(_JS_SRCS_RAW "$<GENEX_EVAL:$<TARGET_PROPERTY:${_RULE_WASM_TARGET},IREE_WASM_JS_SRCS>>")
  set(_JS_SRCS "$<FILTER:$<REMOVE_DUPLICATES:${_JS_SRCS_RAW}>,INCLUDE,.+>")

  # Generate the modules manifest at generation time (order-independent).
  # Format: one "module:path" entry per line, read by the bundler.
  string(ASCII 10 _NL)
  file(GENERATE
    OUTPUT "${_MANIFEST_FILE}"
    CONTENT "$<JOIN:${_JS_ENTRIES},${_NL}>"
  )

  # Determine the entry point.
  if(_RULE_MAIN)
    # Explicit entry point (iree_wasm_cc_binary with MAIN parameter).
    get_filename_component(_EFFECTIVE_MAIN "${_RULE_MAIN}" ABSOLUTE)
    set(_ENTRY_SRCS "")
  else()
    # Discover entry point from direct deps via generation-time genexes.
    # Each dep is checked for IREE_WASM_ENTRY_MAIN — the first non-empty
    # value wins, falling back to the generic test harness.
    set(_FALLBACK_MAIN "${CMAKE_SOURCE_DIR}/build_tools/wasm/wasm_test_main.mjs")
    set(_EFFECTIVE_MAIN "${_FALLBACK_MAIN}")
    set(_ENTRY_SRCS "")
    foreach(_DEP ${_RULE_DEPS})
      set(_DEP_ENTRY "$<TARGET_PROPERTY:${_DEP},IREE_WASM_ENTRY_MAIN>")
      set(_EFFECTIVE_MAIN "$<IF:$<BOOL:${_DEP_ENTRY}>,${_DEP_ENTRY},${_EFFECTIVE_MAIN}>")
      set(_DEP_SRCS "$<TARGET_PROPERTY:${_DEP},IREE_WASM_ENTRY_SRCS>")
      set(_ENTRY_SRCS "$<IF:$<BOOL:${_DEP_SRCS}>,${_DEP_SRCS},${_ENTRY_SRCS}>")
    endforeach()
  endif()

  # Run the bundler to produce the .mjs bundle.
  add_custom_command(
    OUTPUT "${_RULE_OUTPUT}"
    COMMAND ${Python3_EXECUTABLE} "${_BUNDLER}"
      --wasm "$<TARGET_FILE:${_RULE_WASM_TARGET}>"
      --wasm-filename "$<TARGET_FILE_NAME:${_RULE_WASM_TARGET}>"
      --main "${_EFFECTIVE_MAIN}"
      --modules "${_MANIFEST_FILE}"
      --output "${_RULE_OUTPUT}"
    DEPENDS
      ${_RULE_WASM_TARGET}
      "${_EFFECTIVE_MAIN}"
      "${_BUNDLER}"
      "${_MANIFEST_FILE}"
      ${_ENTRY_SRCS}
      "${_JS_SRCS}"
    COMMENT "Bundling JS companions for ${_RULE_WASM_TARGET}"
    VERBATIM
  )
endfunction()

# iree_wasm_cc_binary()
#
# Creates a wasm binary with bundled JS companions. CMake equivalent of the
# Bazel iree_wasm_cc_binary macro.
#
# This function creates two targets:
#   {name}_wasm: the raw executable producing the wasm binary.
#   {name}: custom target that runs the bundler to produce {name}.mjs.
#
# The bundler walks the transitive deps, collects all iree_wasm_cc_library
# targets, parses the wasm import section for dead code elimination, and
# concatenates the JS companions in dependency order with the entry point.
#
# Parameters:
#   NAME: target name. The output is {name}.mjs.
#   MAIN: entry point JS file. This is the JS equivalent of main.cc — it
#       orchestrates wasm instantiation using the generated
#       createWasmImports() function.
#   SRCS: C/C++ source files for the wasm binary.
#   DEPS: dependencies (both C libraries and iree_wasm_cc_library targets).
#   COPTS: additional compiler options.
#   DEFINES: preprocessor definitions.
#   TESTONLY: whether this is test-only.
function(iree_wasm_cc_binary)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "PACKAGE;NAME;MAIN"
    "SRCS;DEPS;COPTS;DEFINES"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Package naming.
  if(_RULE_PACKAGE)
    set(_PACKAGE_NS "${_RULE_PACKAGE}")
    string(REPLACE "::" "_" _PACKAGE_NAME ${_RULE_PACKAGE})
  else()
    iree_package_ns(_PACKAGE_NS)
    iree_package_name(_PACKAGE_NAME)
  endif()
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  set(_WASM_NAME "${_NAME}_wasm")

  # Resolve dependencies passed by ::name with iree::package::name.
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Create the wasm executable.
  add_executable(${_WASM_NAME} ${_RULE_SRCS})
  if(_RULE_COPTS)
    target_compile_options(${_WASM_NAME} PRIVATE ${_RULE_COPTS})
  endif()
  if(_RULE_DEFINES)
    target_compile_definitions(${_WASM_NAME} PRIVATE ${_RULE_DEFINES})
  endif()
  if(_RULE_DEPS)
    target_link_libraries(${_WASM_NAME} PRIVATE ${_RULE_DEPS})
  endif()

  set(_OUTPUT_MJS "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.mjs")
  _iree_wasm_setup_bundler(
    WASM_TARGET ${_WASM_NAME}
    OUTPUT "${_OUTPUT_MJS}"
    MAIN "${_RULE_MAIN}"
    DEPS ${_RULE_DEPS}
  )

  add_custom_target(${_NAME} ALL DEPENDS "${_OUTPUT_MJS}")
endfunction()

# iree_wasm_cc_test()
#
# Same as iree_wasm_cc_binary, plus a ctest test that runs the bundle via
# Node.js. The test passes if the .mjs entry point exits with code 0.
#
# Parameters: same as iree_wasm_cc_binary.
function(iree_wasm_cc_test)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "PACKAGE;NAME;MAIN"
    "SRCS;DEPS;COPTS;DEFINES"
    ${ARGN}
  )

  # Build the bundle using iree_wasm_cc_binary.
  iree_wasm_cc_binary(${ARGN})

  # Derive the output .mjs path (must match iree_wasm_cc_binary's output).
  set(_OUTPUT_MJS "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.mjs")

  # Register a ctest that runs node on the bundle.
  if(_RULE_PACKAGE)
    set(_PACKAGE_NS "${_RULE_PACKAGE}")
    string(REPLACE "::" "_" _PACKAGE_NAME ${_RULE_PACKAGE})
  else()
    iree_package_ns(_PACKAGE_NS)
    iree_package_name(_PACKAGE_NAME)
  endif()
  set(_TEST_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  find_program(_NODE_EXECUTABLE node REQUIRED)
  add_test(
    NAME "${_TEST_NAME}"
    COMMAND "${_NODE_EXECUTABLE}" "${_OUTPUT_MJS}"
  )
endfunction()
