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
#include <string>
#include <vector>

// IREE's C API:
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/rt/api.h"
#include "iree/vm/api.h"

// Other dependencies (helpers, etc.)
#include "absl/base/macros.h"
#include "absl/types/span.h"
#include "iree/base/logging.h"

// NOTE: order matters here, imgui must come first:
#include "third_party/dear_imgui/imgui.h"
// NOTE: must follow imgui.h:
#include "third_party/dear_imgui/examples/imgui_impl_sdl.h"
#include "third_party/dear_imgui/examples/imgui_impl_vulkan.h"

// Compiled module embedded here to avoid file IO:
#include "iree/samples/rt/vulkan/simple_mul_bytecode_module.h"

static VkAllocationCallbacks* g_Allocator = NULL;
static VkInstance g_Instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_PhysicalDevice = VK_NULL_HANDLE;
static VkDevice g_Device = VK_NULL_HANDLE;
static uint32_t g_QueueFamily = (uint32_t)-1;
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

#define CHECK_IREE_OK(status) CHECK_EQ(IREE_STATUS_OK, (status))

static void SetupVulkan(const char** extensions, uint32_t extensions_count) {
  VkResult err;

  // Create Vulkan Instance
  {
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.enabledExtensionCount = extensions_count;
    create_info.ppEnabledExtensionNames = extensions;
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

  // Select graphics queue family
  {
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, NULL);
    VkQueueFamilyProperties* queues = (VkQueueFamilyProperties*)malloc(
        sizeof(VkQueueFamilyProperties) * count);
    vkGetPhysicalDeviceQueueFamilyProperties(g_PhysicalDevice, &count, queues);
    for (uint32_t i = 0; i < count; i++)
      if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        g_QueueFamily = i;
        break;
      }
    free(queues);
    IM_ASSERT(g_QueueFamily != (uint32_t)-1);
  }

  // Create Logical Device (with 1 queue)
  {
    int device_extension_count = 1;
    const char* device_extensions[] = {"VK_KHR_swapchain"};
    const float queue_priority[] = {1.0f};
    VkDeviceQueueCreateInfo queue_info[1] = {};
    queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info[0].queueFamilyIndex = g_QueueFamily;
    queue_info[0].queueCount = 1;
    queue_info[0].pQueuePriorities = queue_priority;
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount =
        sizeof(queue_info) / sizeof(queue_info[0]);
    create_info.pQueueCreateInfos = queue_info;
    create_info.enabledExtensionCount = device_extension_count;
    create_info.ppEnabledExtensionNames = device_extensions;
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
    pool_info.maxSets = 1000 * ABSL_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)ABSL_ARRAYSIZE(pool_sizes);
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
      (size_t)ABSL_ARRAYSIZE(requestSurfaceImageFormat),
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
      ABSL_ARRAYSIZE(present_modes));

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

extern "C" int main(int argc, char** argv) {
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
      "IREE Samples - Vulkan Graphics Integration", SDL_WINDOWPOS_CENTERED,
      SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);

  // Setup Vulkan
  uint32_t extensions_count = 0;
  SDL_Vulkan_GetInstanceExtensions(window, &extensions_count, NULL);
  const char** extensions = new const char*[extensions_count];
  SDL_Vulkan_GetInstanceExtensions(window, &extensions_count, extensions);
  SetupVulkan(extensions, extensions_count);
  delete[] extensions;

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
  CHECK_IREE_OK(iree_api_init(&argc, &argv));

  // Check API version.
  iree_api_version_t actual_version;
  iree_status_t status =
      iree_api_version_check(IREE_API_VERSION_LATEST, &actual_version);
  if (status != IREE_STATUS_OK) {
    LOG(FATAL) << "Unsupported runtime API version " << actual_version;
  } else {
    LOG(INFO) << "IREE runtime API version " << actual_version;
  }

  // Initialize using Vulkan.
  // TODO(scotttodd): Pass Vulkan device in here and reuse
  iree_rt_instance_t* instance = nullptr;
  CHECK_IREE_OK(iree_rt_instance_create(IREE_ALLOCATOR_DEFAULT, &instance));
  std::string driver_name = "vulkan";
  LOG(INFO) << "Creating driver '" << driver_name << "'";
  CHECK_IREE_OK(iree_rt_instance_register_driver_ex(
      instance, iree_string_view_t{driver_name.data(), driver_name.size()}));
  LOG(INFO) << "Created driver '" << driver_name << "'";

  // Allocate a context that will hold the module state across invocations.
  iree_rt_policy_t* dummy_policy = nullptr;
  CHECK_IREE_OK(iree_rt_policy_create(IREE_ALLOCATOR_DEFAULT, &dummy_policy));
  iree_rt_context_t* context = nullptr;
  CHECK_IREE_OK(iree_rt_context_create(instance, dummy_policy,
                                       IREE_ALLOCATOR_DEFAULT, &context));
  iree_rt_policy_release(dummy_policy);

  // Load bytecode module from embedded data.
  LOG(INFO) << "Loading simple_mul.mlir...";
  const auto* module_file_toc =
      iree::rt::samples::simple_mul_bytecode_module_create();
  iree_rt_module_t* bytecode_module = nullptr;
  CHECK_IREE_OK(iree_vm_bytecode_module_create_from_buffer(
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          module_file_toc->size},
      nullptr, nullptr, IREE_ALLOCATOR_DEFAULT, &bytecode_module));

  // Register bytecode module within the context.
  std::vector<iree_rt_module_t*> modules;
  modules.push_back(bytecode_module);
  CHECK_IREE_OK(
      iree_rt_context_register_modules(context, &modules[0], modules.size()));
  iree_rt_module_release(bytecode_module);
  LOG(INFO) << "Module loaded and context is ready for use";

  // Lookup the entry point function.
  iree_rt_function_t main_function;
  const char kMainFunctionName[] = "module.simple_mul";
  CHECK_IREE_OK(iree_rt_context_resolve_function(
      context,
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

        // Allocate buffers that can be mapped on the CPU and that can also be
        // used on the device. Not all devices support this, but the ones we
        // have now do.
        DLOG(INFO) << "Creating I/O buffers...";
        constexpr int kElementCount = 4;
        iree_hal_buffer_t* arg0_buffer = nullptr;
        iree_hal_buffer_t* arg1_buffer = nullptr;
        CHECK_IREE_OK(iree_rt_context_allocate_device_visible_buffer(
            context, IREE_HAL_BUFFER_USAGE_ALL, sizeof(float) * kElementCount,
            IREE_ALLOCATOR_DEFAULT, &arg0_buffer));
        CHECK_IREE_OK(iree_rt_context_allocate_device_visible_buffer(
            context, IREE_HAL_BUFFER_USAGE_ALL, sizeof(float) * kElementCount,
            IREE_ALLOCATOR_DEFAULT, &arg1_buffer));

        // Write inputs into the mappable buffers.
        CHECK_IREE_OK(iree_hal_buffer_write_data(arg0_buffer, 0, &input_x,
                                                 sizeof(input_x)));
        CHECK_IREE_OK(iree_hal_buffer_write_data(arg1_buffer, 0, &input_y,
                                                 sizeof(input_y)));

        // Wrap buffers in buffer views to provide shape information.
        std::array<iree_hal_buffer_view_t*, 2> arg_buffer_views;
        CHECK_IREE_OK(iree_hal_buffer_view_create(
            arg0_buffer, iree_shape_t{1, {kElementCount}}, sizeof(float),
            IREE_ALLOCATOR_DEFAULT, &arg_buffer_views[0]));
        CHECK_IREE_OK(iree_hal_buffer_view_create(
            arg1_buffer, iree_shape_t{1, {kElementCount}}, sizeof(float),
            IREE_ALLOCATOR_DEFAULT, &arg_buffer_views[1]));
        iree_hal_buffer_release(arg0_buffer);
        iree_hal_buffer_release(arg1_buffer);

        // Call into the @simple_mul function.
        DLOG(INFO) << "Calling @simple_mul...";
        iree_rt_invocation_t* invocation = nullptr;
        CHECK_IREE_OK(iree_rt_invocation_create(
            context, &main_function, nullptr, nullptr, arg_buffer_views.data(),
            2, nullptr, 0, IREE_ALLOCATOR_DEFAULT, &invocation));
        CHECK_IREE_OK(iree_hal_buffer_view_release(arg_buffer_views[0]));
        CHECK_IREE_OK(iree_hal_buffer_view_release(arg_buffer_views[1]));
        // TODO(scotttodd): Make this async. Poll each render frame?
        CHECK_IREE_OK(
            iree_rt_invocation_await(invocation, IREE_TIME_INFINITE_FUTURE));

        // Get the result buffers from the invocation.
        DLOG(INFO) << "Retrieving results...";
        std::array<iree_hal_buffer_view_t*, 2> result_buffer_views;
        iree_host_size_t result_count;
        CHECK_IREE_OK(iree_rt_invocation_consume_results(
            invocation, result_buffer_views.size(), IREE_ALLOCATOR_DEFAULT,
            result_buffer_views.data(), &result_count));
        iree_rt_invocation_release(invocation);

        // Read back the results.
        DLOG(INFO) << "Reading back results...";
        iree_hal_buffer_t* result_buffer =
            iree_hal_buffer_view_buffer(result_buffer_views[0]);
        iree_hal_mapped_memory_t mapped_memory;
        CHECK_IREE_OK(iree_hal_buffer_map(result_buffer,
                                          IREE_HAL_MEMORY_ACCESS_READ, 0,
                                          IREE_WHOLE_BUFFER, &mapped_memory));
        memcpy(&latest_output, mapped_memory.contents.data,
               mapped_memory.contents.data_length);
        CHECK_IREE_OK(iree_hal_buffer_unmap(result_buffer, &mapped_memory));

        iree_hal_buffer_view_release(result_buffer_views[0]);

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
  iree_rt_context_release(context);
  iree_rt_instance_release(instance);

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
