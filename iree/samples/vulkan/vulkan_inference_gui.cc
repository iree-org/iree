// Copyright 2019 Google LLC
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

// Vulkan Graphics + IREE API Integration Sample.

// IREE's Vulkan HAL is built with VK_NO_PROTOTYPES so Vulkan can be loaded
// dynamically. This sample links against the Vulkan SDK statically, so we
// want prototypes to be included.
#undef VK_NO_PROTOTYPES

#include <SDL.h>
#include <SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstring>
#include <set>
#include <vector>

// IREE's C API:
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/vulkan/api.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref_cc.h"

// Other dependencies (helpers, etc.)
#include "absl/base/macros.h"
#include "absl/types/span.h"
#include "iree/base/logging.h"
#include "iree/base/main.h"

// NOTE: order matters here, imgui must come first:
#include "third_party/dear_imgui/imgui.h"
// NOTE: must follow imgui.h:
#include "third_party/dear_imgui/examples/imgui_impl_sdl.h"
#include "third_party/dear_imgui/examples/imgui_impl_vulkan.h"

// Compiled module embedded here to avoid file IO:
#include "iree/samples/vulkan/simple_mul_bytecode_module.h"

static VkAllocationCallbacks* g_Allocator = NULL;
static VkInstance g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice g_Device = VK_NULL_HANDLE;
static uint32_t g_QueueFamily = (uint32_t)-1;
static VkDeviceQueueCreateInfo g_QueueInfos[1] = {};
static VkQueue g_Queue = VK_NULL_HANDLE;
static VkPipelineCache g_PipelineCache = VK_NULL_HANDLE;
static VkDescriptorPool g_DescriptorPool = VK_NULL_HANDLE;

static ImGui_ImplVulkanH_Window g_MainWindowData;
static uint32_t g_MinImageCount = 2;
static bool g_SwapChainRebuild = false;
static int g_SwapChainResizeWidth = 0;
static int g_SwapChainResizeHeight = 0;

static void check_vk_result(VkResult err) {
  if (err == 0) return;
  LOG(FATAL) << "VkResult: " << err;
}

static std::vector<const char*> GetIreeLayers(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features) {
  iree_host_size_t required_count;
  iree_hal_vulkan_get_layers(extensibility_set, features, 0, NULL,
                             &required_count);
  std::vector<const char*> layers(required_count);
  iree_hal_vulkan_get_layers(extensibility_set, features, layers.size(),
                             layers.data(), &required_count);
  return layers;
}

static std::vector<const char*> GetInstanceLayers(
    iree_hal_vulkan_features_t vulkan_features) {
  // Query the layers that IREE wants / needs.
  std::vector<const char*> required_layers =
      GetIreeLayers(IREE_HAL_VULKAN_INSTANCE_REQUIRED, vulkan_features);
  std::vector<const char*> optional_layers =
      GetIreeLayers(IREE_HAL_VULKAN_INSTANCE_OPTIONAL, vulkan_features);

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
      LOG(FATAL) << "Required layer " << layer_name << " not available";
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

std::vector<const char*> GetIreeExtensions(
    iree_hal_vulkan_extensibility_set_t extensibility_set,
    iree_hal_vulkan_features_t features) {
  iree_host_size_t required_count;
  iree_hal_vulkan_get_extensions(extensibility_set, features, 0, NULL,
                                 &required_count);
  std::vector<const char*> extensions(required_count);
  iree_hal_vulkan_get_extensions(extensibility_set, features, extensions.size(),
                                 extensions.data(), &required_count);
  return extensions;
}

static std::vector<const char*> GetInstanceExtensions(
    SDL_Window* window, iree_hal_vulkan_features_t vulkan_features) {
  // Ask SDL for its list of required instance extensions.
  uint32_t sdl_extensions_count = 0;
  SDL_Vulkan_GetInstanceExtensions(window, &sdl_extensions_count, NULL);
  std::vector<const char*> sdl_extensions(sdl_extensions_count);
  SDL_Vulkan_GetInstanceExtensions(window, &sdl_extensions_count,
                                   sdl_extensions.data());

  std::vector<const char*> iree_required_extensions =
      GetIreeExtensions(IREE_HAL_VULKAN_INSTANCE_REQUIRED, vulkan_features);
  std::vector<const char*> iree_optional_extensions =
      GetIreeExtensions(IREE_HAL_VULKAN_INSTANCE_OPTIONAL, vulkan_features);

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

static std::vector<const char*> GetDeviceExtensions(
    iree_hal_vulkan_features_t vulkan_features) {
  std::vector<const char*> iree_required_extensions =
      GetIreeExtensions(IREE_HAL_VULKAN_DEVICE_REQUIRED, vulkan_features);
  std::vector<const char*> iree_optional_extensions =
      GetIreeExtensions(IREE_HAL_VULKAN_DEVICE_OPTIONAL, vulkan_features);

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

static void SetupVulkan(iree_hal_vulkan_features_t vulkan_features,
                        const char** instance_layers,
                        uint32_t instance_layers_count,
                        const char** instance_extensions,
                        uint32_t instance_extensions_count) {
  VkResult err;

  // Create Vulkan Instance
  {
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.enabledLayerCount = instance_layers_count;
    create_info.ppEnabledLayerNames = instance_layers;
    create_info.enabledExtensionCount = instance_extensions_count;
    create_info.ppEnabledExtensionNames = instance_extensions;
    err = vkCreateInstance(&create_info, g_Allocator, &g_Instance);
    check_vk_result(err);
  }

  // Select GPU
  {
    uint32_t gpu_count;
    err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, NULL);
    check_vk_result(err);
    IM_ASSERT(gpu_count > 0);

    VkPhysicalDevice* gpus =
        (VkPhysicalDevice*)malloc(sizeof(VkPhysicalDevice) * gpu_count);
    err = vkEnumeratePhysicalDevices(g_Instance, &gpu_count, gpus);
    check_vk_result(err);

    // Use the first reported GPU for simplicity.
    g_PhysicalDevice = gpus[0];
    free(gpus);
  }

  // Select queue family. We want a single queue with graphics and compute for
  // simplicity, but we could also discover and use separate queues for each.
  {
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, NULL);
    VkQueueFamilyProperties* queues = (VkQueueFamilyProperties*)malloc(
        sizeof(VkQueueFamilyProperties) * count);
    vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, queues);
    for (uint32_t i = 0; i < count; i++)
      if (queues[i].queueFlags &
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) {
        g_QueueFamily = i;
        break;
      }
    free(queues);
    IM_ASSERT(g_QueueFamily != (uint32_t)-1);
  }

  // Create Logical Device (with 1 queue)
  {
    std::vector<const char*> device_extensions =
        GetDeviceExtensions(vulkan_features);
    const float queue_priority[] = {1.0f};
    g_QueueInfos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    g_QueueInfos[0].queueFamilyIndex = g_QueueFamily;
    g_QueueInfos[0].queueCount = 1;
    g_QueueInfos[0].pQueuePriorities = queue_priority;
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount =
        sizeof(g_QueueInfos) / sizeof(g_QueueInfos[0]);
    create_info.pQueueCreateInfos = g_QueueInfos;
    create_info.enabledExtensionCount = device_extensions.size();
    create_info.ppEnabledExtensionNames = device_extensions.data();
    err =
        vkCreateDevice(g_PhysicalDevice, &create_info, g_Allocator, &g_Device);
    check_vk_result(err);
    vkGetDeviceQueue(g_Device, g_QueueFamily, 0, &g_Queue);
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
    err = vkCreateDescriptorPool(g_Device, &pool_info, g_Allocator,
                                 &g_DescriptorPool);
    check_vk_result(err);
  }
}

static void SetupVulkanWindow(ImGui_ImplVulkanH_Window* wd,
                              VkSurfaceKHR surface, int width, int height) {
  wd->Surface = surface;

  // Check for WSI support
  VkBool32 res;
  vkGetPhysicalDeviceSurfaceSupportKHR(g_PhysicalDevice, g_QueueFamily,
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
      g_PhysicalDevice, wd->Surface, requestSurfaceImageFormat,
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
      g_PhysicalDevice, wd->Surface, &present_modes[0],
      IREE_ARRAYSIZE(present_modes));

  // Create SwapChain, RenderPass, Framebuffer, etc.
  IM_ASSERT(g_MinImageCount >= 2);
  ImGui_ImplVulkanH_CreateWindow(g_Instance, g_PhysicalDevice, g_Device, wd,
                                 g_QueueFamily, g_Allocator, width, height,
                                 g_MinImageCount);

  // Set clear color.
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  memcpy(&wd->ClearValue.color.float32[0], &clear_color, 4 * sizeof(float));
}

static void CleanupVulkan() {
  vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_Allocator);

  vkDestroyDevice(g_Device, g_Allocator);
  vkDestroyInstance(g_Instance, g_Allocator);
}

static void CleanupVulkanWindow() {
  ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData,
                                  g_Allocator);
}

static void FrameRender(ImGui_ImplVulkanH_Window* wd) {
  VkResult err;

  VkSemaphore image_acquired_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
  VkSemaphore render_complete_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
  err = vkAcquireNextImageKHR(g_Device, wd->Swapchain, UINT64_MAX,
                              image_acquired_semaphore, VK_NULL_HANDLE,
                              &wd->FrameIndex);
  check_vk_result(err);

  ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
  {
    err = vkWaitForFences(
        g_Device, 1, &fd->Fence, VK_TRUE,
        UINT64_MAX);  // wait indefinitely instead of periodically checking
    check_vk_result(err);

    err = vkResetFences(g_Device, 1, &fd->Fence);
    check_vk_result(err);
  }
  {
    err = vkResetCommandPool(g_Device, fd->CommandPool, 0);
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
    err = vkQueueSubmit(g_Queue, 1, &info, fd->Fence);
    check_vk_result(err);
  }
}

static void FramePresent(ImGui_ImplVulkanH_Window* wd) {
  VkSemaphore render_complete_semaphore =
      wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
  VkPresentInfoKHR info = {};
  info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  info.waitSemaphoreCount = 1;
  info.pWaitSemaphores = &render_complete_semaphore;
  info.swapchainCount = 1;
  info.pSwapchains = &wd->Swapchain;
  info.pImageIndices = &wd->FrameIndex;
  VkResult err = vkQueuePresentKHR(g_Queue, &info);
  check_vk_result(err);
  wd->SemaphoreIndex =
      (wd->SemaphoreIndex + 1) %
      wd->ImageCount;  // Now we can use the next set of semaphores
}

int iree::IreeMain(int argc, char** argv) {
  // --------------------------------------------------------------------------
  // Create a window.
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
    LOG(FATAL) << "Failed to initialize SDL";
    return 1;
  }

  // Setup window
  SDL_WindowFlags window_flags = (SDL_WindowFlags)(
      SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
  SDL_Window* window = SDL_CreateWindow(
      "IREE Samples - Vulkan Inference GUI", SDL_WINDOWPOS_CENTERED,
      SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);

  // Setup Vulkan
  iree_hal_vulkan_features_t iree_vulkan_features =
      static_cast<iree_hal_vulkan_features_t>(
          IREE_HAL_VULKAN_ENABLE_VALIDATION_LAYERS |
          IREE_HAL_VULKAN_ENABLE_DEBUG_UTILS |
          IREE_HAL_VULKAN_ENABLE_PUSH_DESCRIPTORS);
  std::vector<const char*> layers = GetInstanceLayers(iree_vulkan_features);
  std::vector<const char*> extensions =
      GetInstanceExtensions(window, iree_vulkan_features);
  SetupVulkan(iree_vulkan_features, layers.data(), layers.size(),
              extensions.data(), extensions.size());

  // Create Window Surface
  VkSurfaceKHR surface;
  VkResult err;
  if (SDL_Vulkan_CreateSurface(window, g_Instance, &surface) == 0) {
    printf("Failed to create Vulkan surface.\n");
    return 1;
  }

  // Create Framebuffers
  int w, h;
  SDL_GetWindowSize(window, &w, &h);
  ImGui_ImplVulkanH_Window* wd = &g_MainWindowData;
  SetupVulkanWindow(wd, surface, w, h);

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;

  ImGui::StyleColorsDark();

  // Setup Platform/Renderer bindings
  ImGui_ImplSDL2_InitForVulkan(window);
  ImGui_ImplVulkan_InitInfo init_info = {};
  init_info.Instance = g_Instance;
  init_info.PhysicalDevice = g_PhysicalDevice;
  init_info.Device = g_Device;
  init_info.QueueFamily = g_QueueFamily;
  init_info.Queue = g_Queue;
  init_info.PipelineCache = g_PipelineCache;
  init_info.DescriptorPool = g_DescriptorPool;
  init_info.Allocator = g_Allocator;
  init_info.MinImageCount = g_MinImageCount;
  init_info.ImageCount = wd->ImageCount;
  init_info.CheckVkResultFn = check_vk_result;
  ImGui_ImplVulkan_Init(&init_info, wd->RenderPass);

  // Upload Fonts
  {
    // Use any command queue
    VkCommandPool command_pool = wd->Frames[wd->FrameIndex].CommandPool;
    VkCommandBuffer command_buffer = wd->Frames[wd->FrameIndex].CommandBuffer;

    err = vkResetCommandPool(g_Device, command_pool, 0);
    check_vk_result(err);
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    err = vkBeginCommandBuffer(command_buffer, &begin_info);
    check_vk_result(err);

    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

    VkSubmitInfo end_info = {};
    end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    end_info.commandBufferCount = 1;
    end_info.pCommandBuffers = &command_buffer;
    err = vkEndCommandBuffer(command_buffer);
    check_vk_result(err);
    err = vkQueueSubmit(g_Queue, 1, &end_info, VK_NULL_HANDLE);
    check_vk_result(err);

    err = vkDeviceWaitIdle(g_Device);
    check_vk_result(err);
    ImGui_ImplVulkan_DestroyFontUploadObjects();
  }

  // Demo state.
  bool show_demo_window = true;
  bool show_iree_window = true;
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Setup IREE.
  // This call to |iree_api_init| is not technically required, but it is
  // included for completeness.
  IREE_CHECK_OK(iree_api_init(&argc, &argv));

  // Check API version.
  iree_api_version_t actual_version;
  iree_status_t status =
      iree_api_version_check(IREE_API_VERSION_LATEST, &actual_version);
  if (iree_status_is_ok(status)) {
    LOG(INFO) << "IREE runtime API version " << actual_version;
  } else {
    LOG(FATAL) << "Unsupported runtime API version " << actual_version;
  }

  // Register HAL module types.
  IREE_CHECK_OK(iree_hal_module_register_types());

  // Create a runtime Instance.
  iree_vm_instance_t* iree_instance = nullptr;
  IREE_CHECK_OK(
      iree_vm_instance_create(iree_allocator_system(), &iree_instance));

  // Create IREE Vulkan Driver and Device, sharing our VkInstance/VkDevice.
  LOG(INFO) << "Creating Vulkan driver/device";
  // Load symbols from our static `vkGetInstanceProcAddr` for IREE to use.
  iree_hal_vulkan_syms_t* iree_vk_syms = nullptr;
  IREE_CHECK_OK(iree_hal_vulkan_syms_create(
      reinterpret_cast<void*>(&vkGetInstanceProcAddr), &iree_vk_syms));
  // Create the driver sharing our VkInstance.
  iree_hal_driver_t* iree_vk_driver = nullptr;
  iree_hal_vulkan_driver_options_t options;
  options.api_version = VK_API_VERSION_1_0;
  options.features = static_cast<iree_hal_vulkan_features_t>(
      IREE_HAL_VULKAN_ENABLE_DEBUG_UTILS |
      IREE_HAL_VULKAN_ENABLE_PUSH_DESCRIPTORS);
  IREE_CHECK_OK(iree_hal_vulkan_driver_create_using_instance(
      options, iree_vk_syms, g_Instance, &iree_vk_driver));
  // Create a device sharing our VkDevice and queue.
  // We could also create a separate (possibly low priority) compute queue for
  // IREE, and/or provide a dedicated transfer queue.
  iree_hal_vulkan_queue_set_t compute_queue_set;
  compute_queue_set.queue_family_index = g_QueueFamily;
  compute_queue_set.queue_indices = 1 << 0;
  iree_hal_vulkan_queue_set_t transfer_queue_set;
  transfer_queue_set.queue_indices = 0;
  iree_hal_device_t* iree_vk_device = nullptr;
  IREE_CHECK_OK(iree_hal_vulkan_driver_wrap_device(
      iree_vk_driver, g_PhysicalDevice, g_Device, compute_queue_set,
      transfer_queue_set, &iree_vk_device));
  // Create a HAL module using the HAL device.
  iree_vm_module_t* hal_module = nullptr;
  IREE_CHECK_OK(iree_hal_module_create(iree_vk_device, iree_allocator_system(),
                                       &hal_module));

  // Load bytecode module from embedded data.
  LOG(INFO) << "Loading simple_mul.mlir...";
  const auto* module_file_toc =
      iree::samples::vulkan::simple_mul_bytecode_module_create();
  iree_vm_module_t* bytecode_module = nullptr;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          module_file_toc->size},
      iree_allocator_null(), iree_allocator_system(), &bytecode_module));

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* iree_context = nullptr;
  std::vector<iree_vm_module_t*> modules = {hal_module, bytecode_module};
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      iree_instance, modules.data(), modules.size(), iree_allocator_system(),
      &iree_context));
  LOG(INFO) << "Module loaded and context is ready for use";

  // Lookup the entry point function.
  iree_vm_function_t main_function;
  const char kMainFunctionName[] = "module.simple_mul";
  IREE_CHECK_OK(iree_vm_context_resolve_function(
      iree_context,
      iree_string_view_t{kMainFunctionName, sizeof(kMainFunctionName) - 1},
      &main_function));
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Main loop.
  bool done = false;
  while (!done) {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        done = true;
      }

      ImGui_ImplSDL2_ProcessEvent(&event);
      if (event.type == SDL_QUIT) done = true;
      if (event.type == SDL_WINDOWEVENT &&
          event.window.event == SDL_WINDOWEVENT_RESIZED &&
          event.window.windowID == SDL_GetWindowID(window)) {
        g_SwapChainResizeWidth = (int)event.window.data1;
        g_SwapChainResizeHeight = (int)event.window.data2;
        g_SwapChainRebuild = true;
      }
    }

    if (g_SwapChainRebuild) {
      g_SwapChainRebuild = false;
      ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
      ImGui_ImplVulkanH_CreateWindow(g_Instance, g_PhysicalDevice, g_Device,
                                     &g_MainWindowData, g_QueueFamily,
                                     g_Allocator, g_SwapChainResizeWidth,
                                     g_SwapChainResizeHeight, g_MinImageCount);
      g_MainWindowData.FrameIndex = 0;
    }

    // Start the Dear ImGui frame
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame(window);
    ImGui::NewFrame();

    // Demo window.
    if (show_demo_window) ImGui::ShowDemoWindow(&show_demo_window);

    // Custom window.
    {
      ImGui::Begin("IREE Vulkan Integration Demo", &show_iree_window,
                   ImGuiWindowFlags_AlwaysAutoResize);

      ImGui::Checkbox("Show ImGui Demo Window", &show_demo_window);
      ImGui::Separator();

      // ImGui Inputs for two input tensors.
      // Run computation whenever any of the values changes.
      static bool dirty = true;
      static float input_x[] = {4.0f, 4.0f, 4.0f, 4.0f};
      static float input_y[] = {2.0f, 2.0f, 2.0f, 2.0f};
      static float latest_output[] = {0.0f, 0.0f, 0.0f, 0.0f};
      ImGui::Text("Multiply numbers using IREE");
      ImGui::PushItemWidth(60);
      // clang-format off
      if (ImGui::DragFloat("= x[0]", &input_x[0], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
      if (ImGui::DragFloat("= x[1]", &input_x[1], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
      if (ImGui::DragFloat("= x[2]", &input_x[2], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
      if (ImGui::DragFloat("= x[3]", &input_x[3], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; }                     // NOLINT
      if (ImGui::DragFloat("= y[0]", &input_y[0], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
      if (ImGui::DragFloat("= y[1]", &input_y[1], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
      if (ImGui::DragFloat("= y[2]", &input_y[2], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; } ImGui::SameLine();  // NOLINT
      if (ImGui::DragFloat("= y[3]", &input_y[3], 0.5f, 0.f, 0.f, "%.1f")) { dirty = true; }                     // NOLINT
      // clang-format on
      ImGui::PopItemWidth();

      if (dirty) {
        // Some input values changed, run the computation.
        // This is synchronous and doesn't reuse buffers for now.

        // Write inputs into mappable buffers.
        DLOG(INFO) << "Creating I/O buffers...";
        constexpr int32_t kElementCount = 4;
        iree_hal_allocator_t* allocator =
            iree_hal_device_allocator(iree_vk_device);
        iree_hal_buffer_t* input0_buffer = nullptr;
        iree_hal_buffer_t* input1_buffer = nullptr;
        iree_hal_memory_type_t input_memory_type =
            static_cast<iree_hal_memory_type_t>(
                IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE);
        iree_hal_buffer_usage_t input_buffer_usage =
            static_cast<iree_hal_buffer_usage_t>(
                IREE_HAL_BUFFER_USAGE_ALL | IREE_HAL_BUFFER_USAGE_CONSTANT);
        IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
            allocator, input_memory_type, input_buffer_usage,
            sizeof(float) * kElementCount, &input0_buffer));
        IREE_CHECK_OK(iree_hal_allocator_allocate_buffer(
            allocator, input_memory_type, input_buffer_usage,
            sizeof(float) * kElementCount, &input1_buffer));
        IREE_CHECK_OK(iree_hal_buffer_write_data(input0_buffer, 0, &input_x,
                                                 sizeof(input_x)));
        IREE_CHECK_OK(iree_hal_buffer_write_data(input1_buffer, 0, &input_y,
                                                 sizeof(input_y)));
        // Wrap input buffers in buffer views.
        iree_hal_buffer_view_t* input0_buffer_view = nullptr;
        iree_hal_buffer_view_t* input1_buffer_view = nullptr;
        IREE_CHECK_OK(iree_hal_buffer_view_create(
            input0_buffer, /*shape=*/&kElementCount, /*shape_rank=*/1,
            IREE_HAL_ELEMENT_TYPE_FLOAT_32, iree_allocator_system(),
            &input0_buffer_view));
        IREE_CHECK_OK(iree_hal_buffer_view_create(
            input1_buffer, /*shape=*/&kElementCount, /*shape_rank=*/1,
            IREE_HAL_ELEMENT_TYPE_FLOAT_32, iree_allocator_system(),
            &input1_buffer_view));
        iree_hal_buffer_release(input0_buffer);
        iree_hal_buffer_release(input1_buffer);
        // Marshal input buffer views through a VM variant list.
        vm::ref<iree_vm_list_t> inputs;
        IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr, 2,
                                          iree_allocator_system(), &inputs));
        auto input0_buffer_view_ref =
            iree_hal_buffer_view_move_ref(input0_buffer_view);
        auto input1_buffer_view_ref =
            iree_hal_buffer_view_move_ref(input1_buffer_view);
        IREE_CHECK_OK(
            iree_vm_list_push_ref_move(inputs.get(), &input0_buffer_view_ref));
        IREE_CHECK_OK(
            iree_vm_list_push_ref_move(inputs.get(), &input1_buffer_view_ref));

        // Prepare outputs list to accept results from the invocation.
        vm::ref<iree_vm_list_t> outputs;
        IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr,
                                          kElementCount * sizeof(float),
                                          iree_allocator_system(), &outputs));

        // Synchronously invoke the function.
        IREE_CHECK_OK(iree_vm_invoke(iree_context, main_function,
                                     /*policy=*/nullptr, inputs.get(),
                                     outputs.get(), iree_allocator_system()));

        // Read back the results.
        DLOG(INFO) << "Reading back results...";
        auto* output_buffer_view = reinterpret_cast<iree_hal_buffer_view_t*>(
            iree_vm_list_get_ref_deref(outputs.get(), 0,
                                       iree_hal_buffer_view_get_descriptor()));
        auto* output_buffer = iree_hal_buffer_view_buffer(output_buffer_view);
        iree_hal_mapped_memory_t mapped_memory;
        IREE_CHECK_OK(iree_hal_buffer_map(output_buffer,
                                          IREE_HAL_MEMORY_ACCESS_READ, 0,
                                          IREE_WHOLE_BUFFER, &mapped_memory));
        memcpy(&latest_output, mapped_memory.contents.data,
               mapped_memory.contents.data_length);
        iree_hal_buffer_unmap(output_buffer, &mapped_memory);

        dirty = false;
      }

      // Display the latest computation output.
      ImGui::Text("X * Y = [%f, %f, %f, %f]",
                  latest_output[0],  //
                  latest_output[1],  //
                  latest_output[2],  //
                  latest_output[3]);
      ImGui::Separator();

      // Framerate counter.
      ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                  1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

      ImGui::End();
    }

    // Rendering
    ImGui::Render();
    FrameRender(wd);

    FramePresent(wd);
  }
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Cleanup
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);
  iree_vm_context_release(iree_context);
  iree_hal_device_release(iree_vk_device);
  iree_hal_driver_release(iree_vk_driver);
  iree_hal_vulkan_syms_release(iree_vk_syms);
  iree_vm_instance_release(iree_instance);

  err = vkDeviceWaitIdle(g_Device);
  check_vk_result(err);
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();

  CleanupVulkanWindow();
  CleanupVulkan();

  SDL_DestroyWindow(window);
  SDL_Quit();
  // --------------------------------------------------------------------------

  return 0;
}
