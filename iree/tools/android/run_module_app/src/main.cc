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

#include <android/log.h>
#include <android_native_app_glue.h>

#include <chrono>
#include <thread>

#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "iree/base/status.h"
#include "iree/hal/drivers/init.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/tools/utils/vm_util.h"
#include "iree/vm/api.h"

namespace iree {
namespace {

const char* kAppTag = "iree-run-module";
#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, kAppTag, __VA_ARGS__))
#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, kAppTag, __VA_ARGS__))

const char kModuleFileName[] = "module.vmfb";
const char kEntryFunctionFileName[] = "entry_function.txt";
const char kInputsFileName[] = "inputs.txt";
const char kDriverFileName[] = "driver.txt";

// A struct containing information regarding one IREE VM module invocation.
struct IreeModuleInvocation {
  std::string module;
  std::string entry_function;
  std::string inputs;
  std::string driver;
};

// A class for loading IREE module invocation information from Android apk asset
// files.
class ModuleLoader {
 public:
  explicit ModuleLoader(android_app* app) : app_context_(app) {}
  ~ModuleLoader() = default;

  Status LoadModuleInvocation(IreeModuleInvocation* out_invocation) {
    IreeModuleInvocation invocation = {};
    IREE_RETURN_IF_ERROR(ReadFileAsset(kModuleFileName, &invocation.module));
    IREE_RETURN_IF_ERROR(
        ReadFileAsset(kEntryFunctionFileName, &invocation.entry_function));
    IREE_RETURN_IF_ERROR(ReadFileAsset(kInputsFileName, &invocation.inputs));
    IREE_RETURN_IF_ERROR(ReadFileAsset(kDriverFileName, &invocation.driver));
    *out_invocation = std::move(invocation);
    return OkStatus();
  }

 private:
  // Reads the given asset file and returns its contents.
  Status ReadFileAsset(const char* file_name, std::string* out_contents) {
    out_contents->clear();

    AAssetManager* asset_manager = app_context_->activity->assetManager;
    AAsset* asset =
        AAssetManager_open(asset_manager, file_name, AASSET_MODE_BUFFER);
    if (!asset) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "failed to open file '%s' in assets",
                              kModuleFileName);
    }

    size_t size_in_bytes = AAsset_getLength(asset);
    std::string contents;
    contents.resize(size_in_bytes);

    AAsset_read(asset, const_cast<char*>(contents.data()), size_in_bytes);
    AAsset_close(asset);

    *out_contents = std::move(contents);
    return OkStatus();
  }

  android_app* app_context_;
};

Status RunModule(const IreeModuleInvocation& invocation) {
  IREE_RETURN_IF_ERROR(iree_hal_module_register_types(),
                       "registering HAL types");
  iree_vm_instance_t* instance = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance),
      "creating instance");

  iree_vm_module_t* input_module = nullptr;
  IREE_RETURN_IF_ERROR(LoadBytecodeModule(invocation.module, &input_module));

  iree_hal_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(CreateDevice(invocation.driver, &device));
  iree_vm_module_t* hal_module = nullptr;
  IREE_RETURN_IF_ERROR(CreateHalModule(device, &hal_module));

  iree_vm_context_t* context = nullptr;
  // Order matters. The input module will likely be dependent on the hal module.
  std::array<iree_vm_module_t*, 2> modules = {hal_module, input_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
                           instance, modules.data(), modules.size(),
                           iree_allocator_system(), &context),
                       "creating context");

  const std::string& function_name = invocation.entry_function;
  iree_vm_function_t function;
  IREE_RETURN_IF_ERROR(
      input_module->lookup_function(
          input_module->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
          iree_string_view_t{function_name.data(), function_name.size()},
          &function),
      "looking up function '%s'", function_name.c_str());

  IREE_RETURN_IF_ERROR(ValidateFunctionAbi(function));
  std::vector<RawSignatureParser::Description> input_descs;
  IREE_RETURN_IF_ERROR(ParseInputSignature(function, &input_descs));

  absl::InlinedVector<absl::string_view, 4> input_views(
      absl::StrSplit(invocation.inputs, '\n', absl::SkipEmpty()));
  vm::ref<iree_vm_list_t> inputs;
  IREE_RETURN_IF_ERROR(ParseToVariantList(
      input_descs, iree_hal_device_allocator(device), input_views, &inputs));

  std::vector<RawSignatureParser::Description> output_descs;
  IREE_RETURN_IF_ERROR(ParseOutputSignature(function, &output_descs));
  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(/*element_type=*/nullptr,
                                           output_descs.size(),
                                           iree_allocator_system(), &outputs));

  LOGI("Execute @%s", function_name.c_str());
  IREE_RETURN_IF_ERROR(
      iree_vm_invoke(context, function, /*policy=*/nullptr, inputs.get(),
                     outputs.get(), iree_allocator_system()),
      "invoking function '%s'", function_name.c_str());

  std::ostringstream oss;
  IREE_RETURN_IF_ERROR(PrintVariantList(output_descs, outputs.get(), &oss),
                       "printing results");
  LOGI("Execution Result:");
  LOGI("%s", oss.str().c_str());

  inputs.reset();
  outputs.reset();
  iree_vm_module_release(hal_module);
  iree_vm_module_release(input_module);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return OkStatus();
}

void RunModuleAppMain(android_app* app) {
  // Sleep for 2 seconds to allow tools like AGI to perform the necessary
  // initialization.
  // TODO(antiagainst): This can be improved by rendering some UI button to
  // trigger the workload.
  std::this_thread::sleep_for(std::chrono::seconds(2));

  IREE_CHECK_OK(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));

  ModuleLoader loader(app);
  IreeModuleInvocation invocation;
  auto status = loader.LoadModuleInvocation(&invocation);
  if (status.ok()) {
    LOGI("entry function: '%s'", invocation.entry_function.c_str());
    LOGI("inputs:\n%s", invocation.inputs.c_str());
    LOGI("driver: '%s'", invocation.driver.c_str());
    status = RunModule(invocation);
    if (!status.ok()) LOGE("%s", status.ToString().c_str());
  } else {
    LOGE("failed to load module invocation: %s", status.ToString().c_str());
  }
}

void HandleAndroidAppCommand(android_app* app, int32_t cmd) {
  switch (cmd) {
    case APP_CMD_INIT_WINDOW:
      RunModuleAppMain(app);
      break;
    default:
      break;
  }
}

}  // namespace
}  // namespace iree

#define NATIVE_EXPORT extern "C" __attribute__((visibility("default")))

// The main entry point of a native application using android_native_app_glue.
// It runs in its own thread with its own event loop.
NATIVE_EXPORT void android_main(struct android_app* app) {
  // Set the callback to process system events.
  app->onAppCmd = iree::HandleAndroidAppCommand;

  int events;
  android_poll_source* source;

  // Main loop for processing events.
  while (app->destroyRequested == 0) {
    if (ALooper_pollAll(/*timeoutMillis=*/1, /*outFd=*/nullptr, &events,
                        (void**)&source) >= 0) {
      if (source != nullptr) source->process(app, source);
    }
  }
}
