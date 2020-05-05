# Copyright 2019 Google LLC
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

set(IMGUI_ROOT "${IREE_ROOT_DIR}/third_party/dear_imgui/")

external_cc_library(
  PACKAGE
    dear_imgui
  NAME
    dear_imgui
  ROOT
    ${IMGUI_ROOT}
  SRCS
    "imgui.cpp"
    "imgui_demo.cpp"
    "imgui_draw.cpp"
    "imgui_widgets.cpp"
  HDRS
    "imconfig.h"
    "imgui.h"
    "imgui_internal.h"
    "imstb_rectpack.h"
    "imstb_textedit.h"
    "imstb_truetype.h"
)

external_cc_library(
  PACKAGE
    dear_imgui
  NAME
    impl_sdl
  ROOT
    ${IMGUI_ROOT}
  SRCS
    "examples/imgui_impl_sdl.cpp"
  HDRS
    "examples/imgui_impl_sdl.h"
  INCLUDES
    ${IMGUI_ROOT}
  DEPS
    dear_imgui::dear_imgui
    SDL2-static
)

# Dear ImGui's Vulkan implementation requires statically link against Vulkan.
# One way to achieve this is by installing the Vulkan SDK from
# https://vulkan.lunarg.com/.
# If Vulkan is not found, omit these targets (and users must do the same).
include(FindVulkan)
if(NOT Vulkan_FOUND)
  message(NOTICE "Vulkan was not found, skipping Dear ImGui Vulkan support")
else()
  external_cc_library(
    PACKAGE
      dear_imgui
    NAME
      impl_vulkan
    ROOT
      ${IMGUI_ROOT}
    SRCS
      "examples/imgui_impl_vulkan.cpp"
    HDRS
      "examples/imgui_impl_vulkan.h"
    INCLUDES
      ${IMGUI_ROOT}
    DEPS
      dear_imgui::dear_imgui
      Vulkan::Vulkan
  )

  add_executable(dear_imgui_example_sdl_vulkan
    ${IMGUI_ROOT}/examples/example_sdl_vulkan/main.cpp
  )
  target_include_directories(dear_imgui_example_sdl_vulkan
    PUBLIC
      ${IMGUI_ROOT}/examples/
  )
  target_link_libraries(dear_imgui_example_sdl_vulkan
    PRIVATE
      dear_imgui::dear_imgui
      dear_imgui::impl_sdl
      dear_imgui::impl_vulkan
  )
endif()
