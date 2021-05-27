// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/vulkan/vulkan_gui_util.h"

#include <cstring>
#include <set>

#include "iree/base/api.h"
#include "iree/base/logging.h"

namespace iree {

namespace {

void check_vk_result(VkResult err) {
  if (err == 0) return;
  IREE_LOG(FATAL) << "VkResult: " << err;
}

// Returns the names of the Vulkan layers used for the given IREE
// |extensibility_set| and |features|.
std::vector<const char*> GetIreeLayers(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features) {
  iree_host_size_t required_count;
  iree_hal_vulkan_query_extensibility_set(
      features, extensibility_set, /*string_capacity=*/0,
      /*out_string_values=*/NULL, &required_count);
  std::vector<const char*> layers(required_count);
  iree_hal_vulkan_query_extensibility_set(features, extensibility_set,
                                          layers.size(), layers.data(),
                                          &required_count);
  return layers;
}

// Returns the names of the Vulkan extensions used for the given IREE
// |extensibility_set| and |features|.
std::vector<const char*> GetIreeExtensions(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features) {
  iree_host_size_t required_count;
  iree_hal_vulkan_query_extensibility_set(
      features, extensibility_set, /*string_capacity=*/0,
      /*out_string_values=*/NULL, &required_count);
  std::vector<const char*> extensions(required_count);
  iree_hal_vulkan_query_extensibility_set(features, extensibility_set,
                                          extensions.size(), extensions.data(),
                                          &required_count);
  return extensions;
}

// Returns the names of the Vulkan extensions used for the given IREE
// |vulkan_features|.
std::vector<const char*> GetDeviceExtensions(
    iree_hal_vulkan_features_t vulkan_features) {
  std::vector<const char*> iree_required_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_REQUIRED,
      vulkan_features);
  std::vector<const char*> iree_optional_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_DEVICE_EXTENSIONS_OPTIONAL,
      vulkan_features);

  // Merge extensions lists, including optional and required for simplicity.
  std::set<const char*> ext_set;
  ext_set.insert("VK_KHR_swapchain");
  ext_set.insert(iree_required_extensions.begin(),
                 iree_required_extensions.end());
  ext_set.insert(iree_optional_extensions.begin(),
                 iree_optional_extensions.end());
  std::vector<const char*> extensions(ext_set.begin(), ext_set.end());
  return extensions;
}

}  // namespace

std::vector<const char*> GetInstanceLayers(
    iree_hal_vulkan_features_t vulkan_features) {
  // Query the layers that IREE wants / needs.
  std::vector<const char*> required_layers = GetIreeLayers(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_REQUIRED, vulkan_features);
  std::vector<const char*> optional_layers = GetIreeLayers(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_LAYERS_OPTIONAL, vulkan_features);

  // Query the layers that are available on the Vulkan ICD.
  uint32_t layer_property_count = 0;
  check_vk_result(
      vkEnumerateInstanceLayerProperties(&layer_property_count, NULL));
  std::vector<VkLayerProperties> layer_properties(layer_property_count);
  check_vk_result(vkEnumerateInstanceLayerProperties(&layer_property_count,
                                                     layer_properties.data()));

  // Match between optional/required and available layers.
  std::vector<const char*> layers;
  for (const char* layer_name : required_layers) {
    bool found = false;
    for (const auto& layer_property : layer_properties) {
      if (std::strcmp(layer_name, layer_property.layerName) == 0) {
        found = true;
        layers.push_back(layer_name);
        break;
      }
    }
    if (!found) {
      IREE_LOG(FATAL) << "Required layer " << layer_name << " not available";
    }
  }
  for (const char* layer_name : optional_layers) {
    for (const auto& layer_property : layer_properties) {
      if (std::strcmp(layer_name, layer_property.layerName) == 0) {
        layers.push_back(layer_name);
        break;
      }
    }
  }

  return layers;
}

std::vector<const char*> GetInstanceExtensions(
    SDL_Window* window, iree_hal_vulkan_features_t vulkan_features) {
  // Ask SDL for its list of required instance extensions.
  uint32_t sdl_extensions_count = 0;
  SDL_Vulkan_GetInstanceExtensions(window, &sdl_extensions_count, NULL);
  std::vector<const char*> sdl_extensions(sdl_extensions_count);
  SDL_Vulkan_GetInstanceExtensions(window, &sdl_extensions_count,
                                   sdl_extensions.data());

  std::vector<const char*> iree_required_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_REQUIRED,
      vulkan_features);
  std::vector<const char*> iree_optional_extensions = GetIreeExtensions(
      IREE_HAL_VULKAN_EXTENSIBILITY_INSTANCE_EXTENSIONS_OPTIONAL,
      vulkan_features);

  // Merge extensions lists, including optional and required for simplicity.
  std::set<const char*> ext_set;
  ext_set.insert(sdl_extensions.begin(), sdl_extensions.end());
  ext_set.insert(iree_required_extensions.begin(),
                 iree_required_extensions.end());
  ext_set.insert(iree_optional_extensions.begin(),
                 iree_optional_extensions.end());
  std::vector<const char*> extensions(ext_set.begin(), ext_set.end());
  return extensions;
}

void SetupVulkan(iree_hal_vulkan_features_t vulkan_features,
                 const char** instance_layers, uint32_t instance_layers_count,
                 const char** instance_extensions,
                 uint32_t instance_extensions_count,
                 const VkAllocationCallbacks* allocator, VkInstance* instance,
                 uint32_t* queue_family_index,
                 VkPhysicalDevice* physical_device, VkQueue* queue,
                 VkDevice* device, VkDescriptorPool* descriptor_pool) {
  VkResult err;

  // Create Vulkan Instance
  {
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.enabledLayerCount = instance_layers_count;
    create_info.ppEnabledLayerNames = instance_layers;
    create_info.enabledExtensionCount = instance_extensions_count;
    create_info.ppEnabledExtensionNames = instance_extensions;
    err = vkCreateInstance(&create_info, allocator, instance);
    check_vk_result(err);
  }

  // Select GPU
  {
    uint32_t gpu_count;
    err = vkEnumeratePhysicalDevices(*instance, &gpu_count, NULL);
    check_vk_result(err);
    IM_ASSERT(gpu_count > 0);

    VkPhysicalDevice* gpus =
        (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * gpu_count);
    err = vkEnumeratePhysicalDevices(*instance, &gpu_count, gpus);
    check_vk_result(err);

    // Use the first reported GPU for simplicity.
    *physical_device = gpus[0];
    free(gpus);
  }

  // Select queue family. We want a single queue with graphics and compute for
  // simplicity, but we could also discover and use separate queues for each.
  {
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(*physical_device, &count, NULL);
    VkQueueFamilyProperties* queues = (VkQueueFamilyProperties*)malloc(
        sizeof(VkQueueFamilyProperties) * count);
    vkGetPhysicalDeviceQueueFamilyProperties(*physical_device, &count, queues);
    for (uint32_t i = 0; i < count; i++) {
      if (queues[i].queueFlags &
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
        *queue_family_index = i;
        break;
      }
    }
    free(queues);
    IM_ASSERT(*queue_family_index != (uint32_t)-1);
  }

  // Create Logical Device (with 1 queue)
  {
    std::vector<const char*> device_extensions =
        GetDeviceExtensions(vulkan_features);
    const float queue_priority[] = {1.0f};
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = *queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = queue_priority;
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_info;
    create_info.enabledExtensionCount =
        static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();
    err = vkCreateDevice(*physical_device, &create_info, allocator, device);
    check_vk_result(err);
    vkGetDeviceQueue(*device, *queue_family_index, 0, queue);
  }

  // Create Descriptor Pool
  {
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1000 * IREE_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IREE_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    err =
        vkCreateDescriptorPool(*device, &pool_info, allocator, descriptor_pool);
    check_vk_result(err);
  }
}

void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd,
                       const VkAllocationCallbacks* allocator,
                       VkInstance instance, uint32_t queue_family_index,
                       VkPhysicalDevice physical_device, VkDevice device,
                       VkSurfaceKHR surface, int width, int height,
                       uint32_t min_image_count) {
  wd->Surface = surface;

  // Check for WSI support
  VkBool32 res;
  vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family_index,
                                       wd->Surface, &res);
  if (res != VK_TRUE) {
    fprintf(stderr, "Error no WSI support on physical device 0\n");
    exit(-1);
  }

  // Select Surface Format
  const VkFormat requestSurfaceImageFormat[] = {
      VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM};
  const VkColorSpaceKHR requestSurfaceColorSpace =
      VK_COLORSPACE_SRGB_NONLINEAR_KHR;
  wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
      physical_device, wd->Surface, requestSurfaceImageFormat,
      (size_t)IREE_ARRAYSIZE(requestSurfaceImageFormat),
      requestSurfaceColorSpace);

  // Select Present Mode
#ifdef IMGUI_UNLIMITED_FRAME_RATE
  VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_MAILBOX_KHR,
                                      VK_PRESENT_MODE_IMMEDIATE_KHR,
                                      VK_PRESENT_MODE_FIFO_KHR};
#else
  VkPresentModeKHR present_modes[] = {VK_PRESENT_MODE_FIFO_KHR};
#endif
  wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
      physical_device, wd->Surface, &present_modes[0],
      IREE_ARRAYSIZE(present_modes));

  // Create SwapChain, RenderPass, Framebuffer, etc.
  IM_ASSERT(min_image_count >= 2);
  ImGui_ImplVulkanH_CreateOrResizeWindow(instance, physical_device, device, wd,
                                         queue_family_index, allocator, width,
                                         height, min_image_count);

  // Set clear color.
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  memcpy(&wd->ClearValue.color.float32[0], &clear_color, 4 * sizeof(float));
}

void RenderFrame(ImGui_ImplVulkanH_Window* wd, VkDevice device, VkQueue queue) {
  VkResult err;

  VkSemaphore image_acquired_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
  VkSemaphore render_complete_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
  err = vkAcquireNextImageKHR(device, wd->Swapchain, UINT64_MAX,
                              image_acquired_semaphore, VK_NULL_HANDLE,
                              &wd->FrameIndex);
  check_vk_result(err);

  ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
  {
    err = vkWaitForFences(
        device, 1, &fd->Fence, VK_TRUE,
        UINT64_MAX);  // wait indefinitely instead of periodically checking
    check_vk_result(err);

    err = vkResetFences(device, 1, &fd->Fence);
    check_vk_result(err);
  }
  {
    err = vkResetCommandPool(device, fd->CommandPool, 0);
    check_vk_result(err);
    VkCommandBufferBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
    check_vk_result(err);
  }
  {
    VkRenderPassBeginInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    info.renderPass = wd->RenderPass;
    info.framebuffer = fd->Framebuffer;
    info.renderArea.extent.width = wd->Width;
    info.renderArea.extent.height = wd->Height;
    info.clearValueCount = 1;
    info.pClearValues = &wd->ClearValue;
    vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
  }

  // Record Imgui Draw Data and draw funcs into command buffer
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), fd->CommandBuffer);

  // Submit command buffer
  vkCmdEndRenderPass(fd->CommandBuffer);
  {
    VkPipelineStageFlags wait_stage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &image_acquired_semaphore;
    info.pWaitDstStageMask = &wait_stage;
    info.commandBufferCount = 1;
    info.pCommandBuffers = &fd->CommandBuffer;
    info.signalSemaphoreCount = 1;
    info.pSignalSemaphores = &render_complete_semaphore;

    err = vkEndCommandBuffer(fd->CommandBuffer);
    check_vk_result(err);
    err = vkQueueSubmit(queue, 1, &info, fd->Fence);
    check_vk_result(err);
  }
}

void PresentFrame(ImGui_ImplVulkanH_Window* wd, VkQueue queue) {
  VkSemaphore render_complete_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
  VkPresentInfoKHR info = {};
  info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  info.waitSemaphoreCount = 1;
  info.pWaitSemaphores = &render_complete_semaphore;
  info.swapchainCount = 1;
  info.pSwapchains = &wd->Swapchain;
  info.pImageIndices = &wd->FrameIndex;
  VkResult err = vkQueuePresentKHR(queue, &info);
  check_vk_result(err);
  wd->SemaphoreIndex =
      (wd->SemaphoreIndex + 1) %
      wd->ImageCount;  // Now we can use the next set of semaphores
}

}  // namespace iree
