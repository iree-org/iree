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

// Vulkan GUI utility functions
// Other matters here: we need to pull in this first to make sure Vulkan API
// prototypes are defined so that we can statically link against them.
#include "iree/testing/vulkan/vulkan_gui_util.h"

// IREE's C API:
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/vulkan/api.h"
#include "iree/hal/vulkan/registration/driver_module.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref_cc.h"

// Other dependencies (helpers, etc.)
#include "absl/base/macros.h"
#include "absl/types/span.h"
#include "iree/base/internal/main.h"
#include "iree/base/logging.h"

// Compiled module embedded here to avoid file IO:
#include "iree/samples/vulkan/simple_mul_bytecode_module.h"

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
  IREE_LOG(FATAL) << "VkResult: " << err;
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

namespace iree {

extern "C" int iree_main(int argc, char** argv) {
  // --------------------------------------------------------------------------
  // Create a window.
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
    IREE_LOG(FATAL) << "Failed to initialize SDL";
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
          IREE_HAL_VULKAN_FEATURE_ENABLE_VALIDATION_LAYERS |
          IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS);
  std::vector<const char*> layers = GetInstanceLayers(iree_vulkan_features);
  std::vector<const char*> extensions =
      GetInstanceExtensions(window, iree_vulkan_features);
  SetupVulkan(iree_vulkan_features, layers.data(),
              static_cast<uint32_t>(layers.size()), extensions.data(),
              static_cast<uint32_t>(extensions.size()), g_Allocator,
              &g_Instance, &g_QueueFamily, &g_PhysicalDevice, &g_Queue,
              &g_Device, &g_DescriptorPool);

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
  SetupVulkanWindow(wd, g_Allocator, g_Instance, g_QueueFamily,
                    g_PhysicalDevice, g_Device, surface, w, h, g_MinImageCount);

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

  // Check API version.
  iree_api_version_t actual_version;
  iree_status_t status =
      iree_api_version_check(IREE_API_VERSION_LATEST, &actual_version);
  if (iree_status_is_ok(status)) {
    IREE_LOG(INFO) << "IREE runtime API version " << actual_version;
  } else {
    IREE_LOG(FATAL) << "Unsupported runtime API version " << actual_version;
  }

  // Register HAL drivers and VM module types.
  IREE_CHECK_OK(iree_hal_vulkan_driver_module_register(
      iree_hal_driver_registry_default()));
  IREE_CHECK_OK(iree_hal_module_register_types());

  // Create a runtime Instance.
  iree_vm_instance_t* iree_instance = nullptr;
  IREE_CHECK_OK(
      iree_vm_instance_create(iree_allocator_system(), &iree_instance));

  // Create IREE Vulkan Driver and Device, sharing our VkInstance/VkDevice.
  IREE_LOG(INFO) << "Creating Vulkan driver/device";
  // Load symbols from our static `vkGetInstanceProcAddr` for IREE to use.
  iree_hal_vulkan_syms_t* iree_vk_syms = nullptr;
  IREE_CHECK_OK(iree_hal_vulkan_syms_create(
      reinterpret_cast<void*>(&vkGetInstanceProcAddr), iree_allocator_system(),
      &iree_vk_syms));
  // Create the driver sharing our VkInstance.
  iree_hal_driver_t* iree_vk_driver = nullptr;
  iree_string_view_t driver_identifier = iree_make_cstring_view("vulkan");
  iree_hal_vulkan_driver_options_t driver_options;
  driver_options.api_version = VK_API_VERSION_1_0;
  driver_options.requested_features = static_cast<iree_hal_vulkan_features_t>(
      IREE_HAL_VULKAN_FEATURE_ENABLE_DEBUG_UTILS);
  IREE_CHECK_OK(iree_hal_vulkan_driver_create_using_instance(
      driver_identifier, &driver_options, iree_vk_syms, g_Instance,
      iree_allocator_system(), &iree_vk_driver));
  // Create a device sharing our VkDevice and queue.
  // We could also create a separate (possibly low priority) compute queue for
  // IREE, and/or provide a dedicated transfer queue.
  iree_string_view_t device_identifier = iree_make_cstring_view("vulkan");
  iree_hal_vulkan_queue_set_t compute_queue_set;
  compute_queue_set.queue_family_index = g_QueueFamily;
  compute_queue_set.queue_indices = 1 << 0;
  iree_hal_vulkan_queue_set_t transfer_queue_set;
  transfer_queue_set.queue_indices = 0;
  iree_hal_device_t* iree_vk_device = nullptr;
  IREE_CHECK_OK(iree_hal_vulkan_wrap_device(
      device_identifier, &driver_options.device_options, iree_vk_syms,
      g_Instance, g_PhysicalDevice, g_Device, &compute_queue_set,
      &transfer_queue_set, iree_allocator_system(), &iree_vk_device));
  // Create a HAL module using the HAL device.
  iree_vm_module_t* hal_module = nullptr;
  IREE_CHECK_OK(iree_hal_module_create(iree_vk_device, iree_allocator_system(),
                                       &hal_module));

  // Load bytecode module from embedded data.
  IREE_LOG(INFO) << "Loading simple_mul.mlir...";
  const auto* module_file_toc =
      iree::samples::vulkan::simple_mul_bytecode_module_create();
  iree_vm_module_t* bytecode_module = nullptr;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file_toc->data),
          module_file_toc->size},
      iree_allocator_null(), iree_allocator_system(), &bytecode_module));
  // Query for details about what is in the loaded module.
  iree_vm_module_signature_t bytecode_module_signature =
      iree_vm_module_signature(bytecode_module);
  IREE_LOG(INFO) << "Module loaded, have <"
                 << bytecode_module_signature.export_function_count
                 << "> exported functions:";
  for (int i = 0; i < bytecode_module_signature.export_function_count; ++i) {
    iree_string_view_t function_name;
    iree_vm_function_signature_t function_signature;
    IREE_CHECK_OK(bytecode_module->get_function(
        bytecode_module->self, IREE_VM_FUNCTION_LINKAGE_EXPORT, i,
        /*out_function=*/nullptr, &function_name, &function_signature));
    IREE_LOG(INFO) << "  " << i << ": '"
                   << std::string(function_name.data, function_name.size)
                   << "' with calling convention '"
                   << std::string(function_signature.calling_convention.data,
                                  function_signature.calling_convention.size)
                   << "'";
  }

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* iree_context = nullptr;
  std::vector<iree_vm_module_t*> modules = {hal_module, bytecode_module};
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      iree_instance, modules.data(), modules.size(), iree_allocator_system(),
      &iree_context));
  IREE_LOG(INFO) << "Context with modules is ready for use";

  // Lookup the async entry point function.
  iree_vm_function_t main_function;
  const char kMainFunctionName[] = "module.simple_mul$async";
  IREE_CHECK_OK(iree_vm_context_resolve_function(
      iree_context,
      iree_string_view_t{kMainFunctionName, sizeof(kMainFunctionName) - 1},
      &main_function));
  iree_string_view_t main_function_name = iree_vm_function_name(&main_function);
  IREE_LOG(INFO) << "Resolved main function named '"
                 << std::string(main_function_name.data,
                                main_function_name.size)
                 << "'";

  // Create wait and signal semaphores for async execution.
  vm::ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_CHECK_OK(
      iree_hal_semaphore_create(iree_vk_device, 0ull, &wait_semaphore));
  vm::ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_CHECK_OK(
      iree_hal_semaphore_create(iree_vk_device, 0ull, &signal_semaphore));
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Main loop.
  bool done = false;
  int frame_number = 0;
  while (!done) {
    frame_number++;
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
      ImGui_ImplVulkanH_CreateOrResizeWindow(
          g_Instance, g_PhysicalDevice, g_Device, &g_MainWindowData,
          g_QueueFamily, g_Allocator, g_SwapChainResizeWidth,
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
        IREE_DLOG(INFO) << "Creating I/O buffers...";
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
            IREE_HAL_ELEMENT_TYPE_FLOAT_32, &input0_buffer_view));
        IREE_CHECK_OK(iree_hal_buffer_view_create(
            input1_buffer, /*shape=*/&kElementCount, /*shape_rank=*/1,
            IREE_HAL_ELEMENT_TYPE_FLOAT_32, &input1_buffer_view));
        iree_hal_buffer_release(input0_buffer);
        iree_hal_buffer_release(input1_buffer);
        // Marshal inputs through a VM variant list.
        // [wait_semaphore|wait_value|arg0|arg1|signal_semaphore|signal_value]
        vm::ref<iree_vm_list_t> inputs;
        IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr, 6,
                                          iree_allocator_system(), &inputs));
        IREE_CHECK_OK(
            iree_vm_list_push_ref_retain(inputs.get(), wait_semaphore));
        iree_vm_value_t wait_value;
        wait_value.type = IREE_VM_VALUE_TYPE_I32;
        wait_value.i32 = 0;
        IREE_CHECK_OK(iree_vm_list_push_value(inputs.get(), &wait_value));
        auto input0_buffer_view_ref =
            iree_hal_buffer_view_move_ref(input0_buffer_view);
        auto input1_buffer_view_ref =
            iree_hal_buffer_view_move_ref(input1_buffer_view);
        IREE_CHECK_OK(
            iree_vm_list_push_ref_move(inputs.get(), &input0_buffer_view_ref));
        IREE_CHECK_OK(
            iree_vm_list_push_ref_move(inputs.get(), &input1_buffer_view_ref));
        IREE_CHECK_OK(
            iree_vm_list_push_ref_retain(inputs.get(), signal_semaphore));
        iree_vm_value_t signal_value;
        signal_value.type = IREE_VM_VALUE_TYPE_I32;
        signal_value.i32 = frame_number;
        IREE_CHECK_OK(iree_vm_list_push_value(inputs.get(), &signal_value));

        // Prepare outputs list to accept results from the invocation.
        vm::ref<iree_vm_list_t> outputs;
        IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr,
                                          kElementCount * sizeof(float),
                                          iree_allocator_system(), &outputs));

        // Asynchronously invoke the function.
        IREE_CHECK_OK(iree_vm_invoke(iree_context, main_function,
                                     /*policy=*/nullptr, inputs.get(),
                                     outputs.get(), iree_allocator_system()));

        // Wait for completion.
        // TODO(scotttodd): Samples showing non-blocking async execution
        //   * poll during update loop
        //   * pipeline execution (use signal from one as wait for another)
        IREE_CHECK_OK(iree_hal_semaphore_wait_with_timeout(
            signal_semaphore.get(), 1, IREE_TIME_INFINITE_FUTURE));

        // Read back the results.
        IREE_DLOG(INFO) << "Reading back results...";
        auto* output_buffer_view = reinterpret_cast<iree_hal_buffer_view_t*>(
            iree_vm_list_get_ref_deref(outputs.get(), 0,
                                       iree_hal_buffer_view_get_descriptor()));
        IREE_CHECK_OK(iree_hal_buffer_read_data(
            iree_hal_buffer_view_buffer(output_buffer_view), 0, latest_output,
            sizeof(latest_output)));

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
    RenderFrame(wd, g_Device, g_Queue);

    PresentFrame(wd, g_Queue);
  }
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Cleanup
  iree_hal_semaphore_release(wait_semaphore.get());
  iree_hal_semaphore_release(signal_semaphore.get());

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

}  // namespace iree
