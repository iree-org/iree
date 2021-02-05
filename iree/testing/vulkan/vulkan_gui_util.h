// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_TESTING_VULKAN_VULKAN_GUI_UTIL_H_
#define IREE_TESTING_VULKAN_VULKAN_GUI_UTIL_H_

#include <SDL.h>
#include <SDL_vulkan.h>
#include <imgui.h>
#include <imgui_impl_sdl.h>
#include <imgui_impl_vulkan.h>
#include <vulkan/vulkan.h>

#include <vector>

#include "iree/hal/vulkan/api.h"

namespace iree {

// Returns the names of the Vulkan instance layers needed for the given IREE
// |vulkan_features|.
std::vector<const char*> GetInstanceLayers(
    iree_hal_vulkan_features_t vulkan_features);

// Returns the names of the Vulkan instance extensions needed for the given IREE
// |vulkan_features|.
std::vector<const char*> GetInstanceExtensions(
    SDL_Window* window, iree_hal_vulkan_features_t vulkan_features);

// Initializes the Vulkan environment with the given |vulkan_features| and
// layers/extensions, and writes various Vulkan handles. If errors occur, this
// function asserts and aborts.
//
// This function creates Vulkan |instance|, selects a GPU and
// |queue_family_index| with both graphics and compute bits, gets the
// |physical_device|, creates a logical |device| from it, and creates a
// |descriptor_pool|.
void SetupVulkan(iree_hal_vulkan_features_t vulkan_features,
                 const char** instance_layers, uint32_t instance_layers_count,
                 const char** instance_extensions,
                 uint32_t instance_extensions_count,
                 const VkAllocationCallbacks* allocator, VkInstance* instance,
                 uint32_t* queue_family_index,
                 VkPhysicalDevice* physical_device, VkQueue* queue,
                 VkDevice* device, VkDescriptorPool* descriptor_pool);

// Sets up a ImGui Vukan GUI window.
//
// This function creates surface, swapchain, framebuffer, and others in
// prepration for rendering.
void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd,
                       const VkAllocationCallbacks* allocator,
                       VkInstance instance, uint32_t queue_family_index,
                       VkPhysicalDevice physical_device, VkDevice device,
                       VkSurfaceKHR surface, int width, int height,
                       uint32_t min_image_count);

// Renders the next frame of the ImGui Vulkan GUI window.
//
// This function acquires next swapchain image, creates a command buffer
// containing a render pass for the next frame, and finally submits to the
// queue.
void RenderFrame(ImGui_ImplVulkanH_Window* wd, VkDevice device, VkQueue queue);

// Presents the next frame of the ImGui Vukan GUI window.
void PresentFrame(ImGui_ImplVulkanH_Window* wd, VkQueue queue);

}  // namespace iree

#endif  // IREE_TESTING_VULKAN_VULKAN_GUI_UTIL_H_
