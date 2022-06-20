# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(MOLTENVK_LIB_PATH "${CMAKE_BINARY_DIR}/third_party/moltenVK/Package/Latest/MoltenVK/MoltenVK.xcframework/macos-arm64_x86_64/libMoltenVK.a")
ExternalProject_Add(project_moltenVK
    GIT_REPOSITORY https://github.com/KhronosGroup/MoltenVK
    GIT_TAG 49f78f91a4883617011059625f79214fe50be940
    BUILD_COMMAND make macos
    CONFIGURE_COMMAND ./fetchDependencies --macos
    INSTALL_COMMAND ""
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third_party/moltenVK"
    BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/moltenVK"
    BUILD_BYPRODUCTS ${MOLTENVK_LIB_PATH}
)

add_library(moltenVK::runtime STATIC IMPORTED GLOBAL)
add_dependencies(moltenVK::runtime project_moltenVK)
set_property(TARGET moltenVK::runtime PROPERTY IMPORTED_LOCATION ${MOLTENVK_LIB_PATH})
