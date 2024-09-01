# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_cmake_installdir "${CMAKE_INSTALL_LIBDIR}/cmake/IREE")

# Generate export sets.
# See runtime/src/CMakeLists.txt
iree_generate_export_targets(
  EXPORT_SET Runtime
  INSTALL_DESTINATION ${_cmake_installdir}
)

# See compiler/bindings/c/CMakeLists.txt
iree_generate_export_targets(
  EXPORT_SET Compiler
  INSTALL_DESTINATION ${_cmake_installdir}
)

set(IREE_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/IREE" CACHE STRING "")
find_prefix_from_config(IREE_CONFIG_CODE IREE_INSTALL_PREFIX "${IREE_INSTALL_PACKAGE_DIR}")
extend_path(IREE_CONFIG_CMAKE_DIR "\${IREE_INSTALL_PREFIX}" "${IREE_INSTALL_PACKAGE_DIR}")

if(IREE_BUILD_COMPILER)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/IREECompilerConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/IREECompilerConfig.cmake
    @ONLY
  )
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/IREECompilerConfig.cmake
    DESTINATION ${_cmake_installdir}
    COMPONENT IREECMakeExports
  )
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/IREERuntimeConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/IREERuntimeConfig.cmake
  @ONLY
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/IREERuntimeConfig.cmake
  DESTINATION ${_cmake_installdir}
  COMPONENT IREECMakeExports
)

file(GLOB _iree_cmake_helpers "${CMAKE_CURRENT_SOURCE_DIR}/iree_*.cmake")
list(APPEND
  _iree_cmake_helpers
  ${CMAKE_CURRENT_SOURCE_DIR}/external_cc_library.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffer_c_library.cmake
)
list(REMOVE_ITEM _iree_cmake_helpers iree_llvm.cmake)

install(FILES
   ${_iree_cmake_helpers}
   DESTINATION ${_cmake_installdir}
   COMPONENT IREECMakeExports
)
