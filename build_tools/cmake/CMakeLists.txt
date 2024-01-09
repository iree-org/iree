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
