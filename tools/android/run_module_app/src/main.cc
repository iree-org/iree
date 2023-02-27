// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <android/log.h>
#include <android_native_app_glue.h>

#include <array>
#include <chrono>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "iree/base/api.h"
#include "iree/modules/hal/module.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

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
const char kDeviceFileName[] = "device.txt";

// A struct containing information regarding one IREE VM module invocation.
struct IreeModuleInvocation {
  std::string module;
  std::string entry_function;
  std::string inputs;
  std::string device;
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
    IREE_RETURN_IF_ERROR(ReadFileAsset(kDeviceFileName, &invocation.device));
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
  iree_vm_instance_t* instance = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance),
      "creating instance");
  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance),
                       "registering HAL types");

  iree_vm_module_t* input_module = nullptr;
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      instance,
      iree_make_const_byte_span((void*)invocation.module.data(),
                                invocation.module.size()),
      iree_allocator_null(), iree_allocator_system(), &input_module));

  iree_hal_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_create_device(
      iree_hal_available_driver_registry(),
      iree_make_string_view(invocation.device.data(), invocation.device.size()),
      iree_allocator_system(), &device));
  iree_vm_module_t* hal_module = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_create(instance, device, IREE_HAL_MODULE_FLAG_NONE,
                             iree_allocator_system(), &hal_module));

  iree_vm_context_t* context = nullptr;
  // Order matters. The input module will likely be dependent on the hal module.
  std::array<iree_vm_module_t*, 2> modules = {hal_module, input_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
                           instance, IREE_VM_CONTEXT_FLAG_NONE, modules.size(),
                           modules.data(), iree_allocator_system(), &context),
                       "creating context");

  const std::string& function_name = invocation.entry_function;
  iree_vm_function_t function;
  IREE_RETURN_IF_ERROR(
      iree_vm_module_lookup_function_by_name(
          input_module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
          iree_string_view_t{function_name.data(), function_name.size()},
          &function),
      "looking up function '%s'", function_name.c_str());

  std::vector<iree_string_view_t> input_views;
  iree_string_view_t inputs_view =
      iree_make_string_view(invocation.inputs.data(), invocation.inputs.size());
  while (!iree_string_view_is_empty(inputs_view)) {
    iree_string_view_t input_view = iree_string_view_empty();
    iree_string_view_split(inputs_view, '\n', &input_view, &inputs_view);
    input_views.push_back(input_view);
  }
  vm::ref<iree_vm_list_t> inputs;
  IREE_RETURN_IF_ERROR(iree_tooling_parse_to_variant_list(
      iree_hal_device_allocator(device), input_views.data(), input_views.size(),
      iree_allocator_system(), &inputs));

  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(/*element_type=*/nullptr, 16,
                                           iree_allocator_system(), &outputs));

  LOGI("Execute @%s", function_name.c_str());
  IREE_RETURN_IF_ERROR(
      iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                     /*policy=*/nullptr, inputs.get(), outputs.get(),
                     iree_allocator_system()),
      "invoking function '%s'", function_name.c_str());

  iree_string_builder_t result_str;
  iree_string_builder_initialize(iree_allocator_system(), &result_str);
  IREE_RETURN_IF_ERROR(
      iree_tooling_append_variant_list_lines(
          outputs.get(), /*max_element_count=*/1024, &result_str),
      "printing results");
  LOGI("Execution Result:");
  LOGI("%.*s", (int)iree_string_builder_size(&result_str),
       iree_string_builder_buffer(&result_str));
  iree_string_builder_deinitialize(&result_str);

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

  ModuleLoader loader(app);
  IreeModuleInvocation invocation;
  auto status = loader.LoadModuleInvocation(&invocation);
  if (status.ok()) {
    LOGI("entry function: '%s'", invocation.entry_function.c_str());
    LOGI("inputs:\n%s", invocation.inputs.c_str());
    LOGI("device: '%s'", invocation.device.c_str());
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
