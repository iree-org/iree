# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT "${IREE_HAL_DRIVER_VULKAN}" OR NOT "${IREE_BUILD_SAMPLES}")
  return()
endif()

# This target statically links against Vulkan.
# One way to achieve this is by installing the Vulkan SDK from
# https://vulkan.lunarg.com/.
# If Vulkan is not found, omit the target.
include(FindVulkan)
if(NOT Vulkan_FOUND)
  message(VERBOSE "Could not find Vulkan, disabling Vulkan GUI programs")
  return()
endif()

# vcpkg install imgui[vulkan-binding,sdl2-binding]
find_package(imgui CONFIG QUIET)
if(NOT imgui_FOUND)
  message(VERBOSE "Could not find Dear ImGui, disabling Vulkan GUI programs")
  return()
endif()

# vcpkg install sdl2[vulkan]
find_package(SDL2 CONFIG QUIET)
if(NOT SDL2_FOUND)
  message(VERBOSE "Could not find SDL2, disabling Vulkan GUI programs")
  return()
endif()

iree_cc_library(
  NAME
    vulkan_gui_util
  HDRS
    "vulkan_gui_util.h"
  SRCS
    "vulkan_gui_util.cc"
  DEPS
    imgui::imgui
    iree::base::api
    iree::base::logging
    iree::hal::vulkan
    SDL2::SDL2
    Vulkan::Vulkan
)

iree_cc_binary(
  NAME
    iree-run-module-vulkan-gui
  SRCS
    "iree-run-module-vulkan-gui-main.cc"
  DEPS
    ::vulkan_gui_util
    absl::flags
    iree::base::internal::file_io
    iree::base::internal::flags
    iree::base::internal::main
    iree::base::status
    iree::base::tracing
    iree::hal::vulkan::registration
    iree::modules::hal
    iree::tools::utils::vm_util
    iree::vm
    iree::vm::bytecode_module
  LINKOPTS
    "${IREE_TARGET_GUI_LINKOPTS}"
)
