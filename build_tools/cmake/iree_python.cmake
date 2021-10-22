# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

###############################################################################
# Main user rules
###############################################################################

# Declares that the current source directory is part of a python package
# that will:
#   - Will create an install target install-COMPONENT (global, not package
#     scoped)
#   - Be installed under python_packages/PACKAGE_NAME
#   - Have a local path of MODULE_PATH (i.e. namespace package path)
#   - Process a setup.py.in from the current directory (if NOT AUGMENT_EXISTING_PACKAGE)
#   - Process a version.py.in from the current directory (if NOT AUGMENT_EXISTING_PACKAGE)
# Will set parent scope variables:
#   - PY_INSTALL_COMPONENT: Install component. Echoed back from the argument
#     for easier addition after this call.
#   - PY_INSTALL_PACKAGES_DIR: The python_packages/PACKAGE_NAME path
#   - PY_INSTALL_MODULE_DIR: The path to the module directory under
#     INSTALL_PACKAGES_DIR.
#
# Add any built deps to DEPS (you will need to add install actions to them
# after).
#
# Any python files in the source directory will be automatically installed
# (recursive).
#
# Also adds a *-stripped target which strips any binaries that are
# present.
#
# Arguments:
#   AUGMENT_EXISTING_PACKAGE: Whether to add install artifacts to an existing
#     package.
#   COMPONENT: Install component
#   PACKAGE_NAME: Name of the Python package in the install directory tree.
#   MODULE_PATH: Relative path within the package to the module being installed.
#   FILES_MATCHING: Explicit arguments to the install FILES_MATCHING directive.
#     (Defaults to "PATTERN *.py")
#   DEPS: Dependencies.
function(iree_py_install_package)
  cmake_parse_arguments(ARG
    "AUGMENT_EXISTING_PACKAGE"
    "COMPONENT;PACKAGE_NAME;MODULE_PATH"
    "DEPS;ADDL_PACKAGE_FILES;FILES_MATCHING"
    ${ARGN})
  set(_install_component ${ARG_COMPONENT})
  set(_install_packages_dir "${CMAKE_INSTALL_PREFIX}/python_packages/${ARG_PACKAGE_NAME}")
  set(_install_module_dir "${_install_packages_dir}/${ARG_MODULE_PATH}")
  set(_target_name install-${_install_component})

  if(NOT FILES_MATCHING)
    set(_files_matching PATTERN "*.py")
  else()
    set(_files_matching ${ARG_FILES_MATCHING})
  endif()

  if(NOT ARG_AUGMENT_EXISTING_PACKAGE)
    configure_file(setup.py.in setup.py)
    install(
      FILES
        ${CMAKE_CURRENT_BINARY_DIR}/setup.py
        ${ARG_ADDL_PACKAGE_FILES}
      COMPONENT ${_install_component}
      DESTINATION "${_install_packages_dir}"
    )
    configure_file(version.py.in version.py)
    install(
      FILES
        ${CMAKE_CURRENT_BINARY_DIR}/version.py
      COMPONENT ${_install_component}
      DESTINATION "${_install_module_dir}"
    )

    set(_component_option -DCMAKE_INSTALL_COMPONENT="${ARG_COMPONENT}")
    add_custom_target(${_target_name}
      COMMAND "${CMAKE_COMMAND}"
              ${_component_option}
              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
      USES_TERMINAL)
    add_custom_target(${_target_name}-stripped
      COMMAND "${CMAKE_COMMAND}"
              ${_component_option}
              -DCMAKE_INSTALL_DO_STRIP=1
              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake"
      USES_TERMINAL)
  endif()

  # Explicit add dependencies in case if we are just extending a package
  # vs adding the targets.
  if(ARG_DEPS)
    add_dependencies(${_target_name} ${ARG_DEPS})
    add_dependencies(${_target_name}-stripped ${ARG_DEPS})
  endif()

  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    COMPONENT ${_install_component}
    DESTINATION "${_install_module_dir}"
    FILES_MATCHING ${_files_matching}
  )

  set(PY_INSTALL_COMPONENT ${_install_component} PARENT_SCOPE)
  set(PY_INSTALL_PACKAGES_DIR "${_install_packages_dir}" PARENT_SCOPE)
  set(PY_INSTALL_MODULE_DIR "${_install_module_dir}" PARENT_SCOPE)
endfunction()

# iree_pyext_module()
#
# Builds a native python module (.so/.dylib/.pyd).
#
# Parameters:
# NAME: name of target
# MODULE_NAME: Base-name of the module.
# SRCS: List of source files for the library
# DEPS: List of other targets the test python libraries require
function(iree_pyext_module)
  cmake_parse_arguments(ARG
    ""
    "NAME;MODULE_NAME;UNIX_LINKER_SCRIPT"
    "SRCS;DEPS;COPTS;INCLUDES"
    ${ARGN})

  iree_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::iree::package::name
  list(TRANSFORM ARG_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  list(TRANSFORM ARG_PYEXT_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  # Prefix the library with the package name, so we get: iree_package_name.
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${ARG_NAME}")

  pybind11_add_module(
    ${_NAME}
    ${ARG_SRCS}
  )

  # Alias the iree_package_name library to iree::package::name so that we can
  # refer to this target with the namespaced format.
  add_library(${_PACKAGE_NS}::${ARG_NAME} ALIAS ${_NAME})

  target_link_libraries(
    ${_NAME}
    PRIVATE ${ARG_DEPS}
  )

  set_target_properties(
    ${_NAME} PROPERTIES
      OUTPUT_NAME "${ARG_MODULE_NAME}"
  )

  target_include_directories(${_NAME}
  PUBLIC
    "$<BUILD_INTERFACE:${ARG_INCLUDES}>"
  )

  # pybind11 requires both RTTI and Exceptions, and it does not know that
  # we have disabled them globally, so turn them back on. Since this is
  # *the only* place in the codebase where we do this, just inline here.
  # Note that this is playing with fire and the extension code is structured
  # so as not to cause problems with RTTI cross-module issues.
  iree_select_compiler_opts(_RTTI_AND_EXCEPTION_COPTS
    CLANG_OR_GCC
      "-frtti"
      "-fexceptions"
    MSVC_OR_CLANG_CL
      # Configure exception handling for standard C++ behavior.
      # - /EHs enables C++ catch-style exceptions
      # - /EHc breaks unwinding across extern C boundaries, dramatically reducing
      #   unwind table size and associated exception handling overhead as the
      #   compiler can assume no exception will ever be thrown within any function
      #   annotated with extern "C".
      # https://docs.microsoft.com/en-us/cpp/build/reference/eh-exception-handling-model
      "/EHsc"
      # Configure RTTI generation.
      # - /GR - Enable generation of RTTI (default)
      # - /GR- - Disables generation of RTTI
      # https://docs.microsoft.com/en-us/cpp/build/reference/gr-enable-run-time-type-information?view=msvc-160
      "/GR"
  )

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  target_compile_options(
    ${_NAME} PRIVATE
    ${ARG_COPTS}
    ${IREE_DEFAULT_COPTS}
    ${_RTTI_AND_EXCEPTION_COPTS}
  )

  # Link flags.
  if(UNIX AND NOT APPLE)  # Apple does not support linker scripts.
    if(ARG_UNIX_LINKER_SCRIPT)
      set_target_properties(${_NAME} PROPERTIES LINK_FLAGS
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/${ARG_UNIX_LINKER_SCRIPT}")
    endif()
  endif()
endfunction()

# iree_py_library()
#
# CMake function to imitate Bazel's iree_py_library rule.
#
# Parameters:
# NAME: name of target
# SRCS: List of source files for the library
# DEPS: List of other targets the test python libraries require
# PYEXT_DEPS: List of deps of extensions built with iree_pyext_module
function(iree_py_library)
  cmake_parse_arguments(
    ARG
    ""
    "NAME"
    "SRCS;DEPS;PYEXT_DEPS"
    ${ARGN}
  )

  iree_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::iree::package::name
  list(TRANSFORM ARG_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${ARG_NAME}")

  add_custom_target(${_NAME} ALL
    DEPENDS ${ARG_DEPS}
  )

  # Symlink each file as its own target.
  foreach(SRC_FILE ${ARG_SRCS})
    # SRC_FILE could have other path components in it, so we need to make a
    # directory for it. Ninja does this automatically, but make doesn't. See
    # https://github.com/google/iree/issues/6801
    set(_SRC_BIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/${SRC_FILE}")
    get_filename_component(_SRC_BIN_DIR "${_SRC_BIN_PATH}" DIRECTORY)
    add_custom_command(
      TARGET ${_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E make_directory "${_SRC_BIN_DIR}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FILE}" "${_SRC_BIN_PATH}"
      BYPRODUCTS "${_SRC_BIN_PATH}"
    )
  endforeach()

  # Add PYEXT_DEPS if any.
  if(ARG_PYEXT_DEPS)
    list(TRANSFORM ARG_PYEXT_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
    add_dependencies(${_NAME} ${ARG_PYEXT_DEPS})
  endif()
endfunction()

# iree_py_test()
#
# CMake function to imitate Bazel's iree_py_test rule.
#
# Parameters:
# NAME: name of test
# SRCS: Test source file (single file only, despite name)
# ARGS: Command line arguments to the Python source file.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# GENERATED_IN_BINARY_DIR: If present, indicates that the srcs have been
#   in the CMAKE_CURRENT_BINARY_DIR.
function(iree_py_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    "GENERATED_IN_BINARY_DIR"
    "NAME;SRCS"
    "ARGS;LABELS"
    ${ARGN}
  )

  # Switch between source and generated tests.
  set(_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(_RULE_GENERATED_IN_BINARY_DIR)
    set(_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}/${_RULE_NAME}")
  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")

  add_test(
    NAME ${_NAME_PATH}
    COMMAND
      "${IREE_SOURCE_DIR}/build_tools/cmake/run_test.${IREE_HOST_SCRIPT_EXT}"
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRCS}"
      ${_RULE_ARGS}
  )

  set_property(TEST ${_NAME_PATH} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_NAME_PATH} PROPERTY ENVIRONMENT
      "PYTHONPATH=${IREE_BINARY_DIR}/compiler-api/python_package:${IREE_BINARY_DIR}/bindings/python:$ENV{PYTHONPATH}"
      "TEST_TMPDIR=${IREE_BINARY_DIR}/tmp/${_NAME}_test_tmpdir"
  )
  iree_add_test_environment_properties(${_NAME_PATH})

  # TODO(marbre): Find out how to add deps to tests.
  #               Similar to _RULE_DATA in iree_lit_test().
endfunction()
