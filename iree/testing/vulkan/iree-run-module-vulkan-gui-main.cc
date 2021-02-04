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

// Vulkan GUI utility functions
// Other matters here: we need to pull in this first to make sure Vulkan API
// prototypes are defined so that we can statically link against them.
#include "iree/testing/vulkan/vulkan_gui_util.h"

// Other dependencies (helpers, etc.)
#include "absl/flags/flag.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/base/internal/main.h"
#include "iree/base/status.h"
#include "iree/hal/vulkan/registration/driver_module.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/tools/utils/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"

ABSL_FLAG(std::string, module_file, "-",
          "File containing the module to load that contains the entry "
          "function. Defaults to stdin.");

ABSL_FLAG(std::string, entry_function, "",
          "Name of a function contained in the module specified by input_file "
          "to run.");

ABSL_FLAG(std::vector<std::string>, function_inputs, {},
          "A comma-separated list of of input buffers of the format:"
          "[shape]xtype=[value]\n"
          "2x2xi32=1 2 3 4\n"
          "Optionally, brackets may be used to separate the element values. "
          "They are ignored by the parser.\n"
          "2x2xi32=[[1 2][3 4]]\n"
          "Due to the absence of repeated flags in absl, commas should not be "
          "used to separate elements. They are reserved for separating input "
          "values:\n"
          "2x2xi32=[[1 2][3 4]], 1x2xf32=[[1 2]]");

ABSL_FLAG(std::string, function_inputs_file, "",
          "Provides a file for input shapes and optional values (see "
          "ParseToVariantListFromFile in vm_util.h for details)");

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

namespace iree {
namespace {

void check_vk_result(VkResult err) {
  if (err == 0) return;
  IREE_LOG(FATAL) << "VkResult: " << err;
}

void CleanupVulkan() {
  vkDestroyDescriptorPool(g_Device, g_DescriptorPool, g_Allocator);

  vkDestroyDevice(g_Device, g_Allocator);
  vkDestroyInstance(g_Instance, g_Allocator);
}

void CleanupVulkanWindow() {
  ImGui_ImplVulkanH_DestroyWindow(g_Instance, g_Device, &g_MainWindowData,
                                  g_Allocator);
}

Status GetModuleContentsFromFlags(std::string* out_contents) {
  auto module_file = absl::GetFlag(FLAGS_module_file);
  if (module_file == "-") {
    *out_contents = std::string{std::istreambuf_iterator<char>(std::cin),
                                std::istreambuf_iterator<char>()};
  } else {
    IREE_RETURN_IF_ERROR(file_io::GetFileContents(module_file, out_contents));
  }
  return OkStatus();
}

// Runs the current IREE bytecode module and renders its result to a window
// using ImGui.
Status RunModuleAndUpdateImGuiWindow(
    iree_hal_device_t* device, iree_vm_context_t* context,
    iree_vm_function_t function, const std::string& function_name,
    const vm::ref<iree_vm_list_t>& function_inputs,
    const std::vector<RawSignatureParser::Description>& output_descs,
    const std::string& window_title) {
  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(/*element_type=*/nullptr,
                                           output_descs.size(),
                                           iree_allocator_system(), &outputs));

  IREE_LOG(INFO) << "EXEC @" << function_name;
  IREE_RETURN_IF_ERROR(iree_vm_invoke(context, function, /*policy=*/nullptr,
                                      function_inputs.get(), outputs.get(),
                                      iree_allocator_system()))
      << "invoking function " << function_name;

  std::ostringstream oss;
  IREE_RETURN_IF_ERROR(PrintVariantList(output_descs, outputs.get(), &oss))
      << "printing results";

  outputs.reset();

  ImGui::Begin(window_title.c_str(), /*p_open=*/nullptr,
               ImGuiWindowFlags_AlwaysAutoResize);

  ImGui::Text("Entry function:");
  ImGui::Text("%s", function_name.c_str());
  ImGui::Separator();

  ImGui::Text("Invocation result:");
  ImGui::Text("%s", oss.str().c_str());
  ImGui::Separator();

  // Framerate counter.
  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
              1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

  ImGui::End();
  return OkStatus();
}
}  // namespace

extern "C" int iree_main(int argc, char** argv) {
  iree_flags_parse_checked(&argc, &argv);
  IREE_CHECK_OK(iree_hal_vulkan_driver_module_register(
      iree_hal_driver_registry_default()));

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
  SetupVulkan(iree_vulkan_features, layers.data(), layers.size(),
              extensions.data(), extensions.size(), g_Allocator, &g_Instance,
              &g_QueueFamily, &g_PhysicalDevice, &g_Queue, &g_Device,
              &g_DescriptorPool);

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

  // Register HAL module types.
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
  // Create a device sharing our VkDevice and queue. This makes capturing with
  // vendor tools easier because we will have sync compute residing in the
  // rendered frame.
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
  IREE_LOG(INFO) << "Loading IREE byecode module...";
  std::string module_file;
  Status status = iree::GetModuleContentsFromFlags(&module_file);
  if (!status.ok()) {
    IREE_LOG(FATAL) << "Error when reading module file" << status;
  }
  iree_vm_module_t* bytecode_module = nullptr;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      iree_const_byte_span_t{
          reinterpret_cast<const uint8_t*>(module_file.data()),
          module_file.size()},
      iree_allocator_null(), iree_allocator_system(), &bytecode_module));

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* iree_context = nullptr;
  std::vector<iree_vm_module_t*> modules = {hal_module, bytecode_module};
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      iree_instance, modules.data(), modules.size(), iree_allocator_system(),
      &iree_context));
  IREE_LOG(INFO) << "Context with modules is ready for use";

  // Lookup the entry point function.
  std::string entry_function = absl::GetFlag(FLAGS_entry_function);
  iree_vm_function_t main_function;
  IREE_CHECK_OK(bytecode_module->lookup_function(
      bytecode_module->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_string_view_t{entry_function.data(), entry_function.size()},
      &main_function));
  iree_string_view_t main_function_name = iree_vm_function_name(&main_function);
  IREE_LOG(INFO) << "Resolved main function named '"
                 << std::string(main_function_name.data,
                                main_function_name.size)
                 << "'";

  IREE_CHECK_OK(ValidateFunctionAbi(main_function));

  std::vector<RawSignatureParser::Description> main_function_input_descs;
  IREE_CHECK_OK(ParseInputSignature(main_function, &main_function_input_descs));
  vm::ref<iree_vm_list_t> main_function_inputs;
  if (!absl::GetFlag(FLAGS_function_inputs_file).empty()) {
    if (!absl::GetFlag(FLAGS_function_inputs).empty()) {
      IREE_LOG(FATAL) << "Expected only one of function_inputs and "
                         "function_inputs_file to be set";
    }
    IREE_CHECK_OK(ParseToVariantListFromFile(
        main_function_input_descs, iree_hal_device_allocator(iree_vk_device),
        absl::GetFlag(FLAGS_function_inputs_file), &main_function_inputs));
  } else {
    IREE_CHECK_OK(ParseToVariantList(
        main_function_input_descs, iree_hal_device_allocator(iree_vk_device),
        absl::GetFlag(FLAGS_function_inputs), &main_function_inputs));
  }

  std::vector<RawSignatureParser::Description> main_function_output_descs;
  IREE_CHECK_OK(
      ParseOutputSignature(main_function, &main_function_output_descs));

  const std::string& window_title = absl::GetFlag(FLAGS_module_file);
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

    // Custom window.
    auto status = RunModuleAndUpdateImGuiWindow(
        iree_vk_device, iree_context, main_function, entry_function,
        main_function_inputs.value(), main_function_output_descs.value(),
        window_title);
    if (!status.ok()) {
      IREE_LOG(FATAL) << status;
      done = true;
      continue;
    }

    // Rendering
    ImGui::Render();
    RenderFrame(wd, g_Device, g_Queue);

    PresentFrame(wd, g_Queue);
  }
  // --------------------------------------------------------------------------

  // --------------------------------------------------------------------------
  // Cleanup
  main_function_inputs.value().reset();

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
