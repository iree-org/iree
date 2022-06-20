# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(APPLE AND IREE_HAL_DRIVER_VULKAN)
  # Add dependencies for Apple SoC GPU/MoltenVK.
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework QuartzCore -framework Metal -framework IOSurface -framework Cocoa -framework IOKit -framework CoreFoundation")
endif()
iree_add_all_subdirs()
