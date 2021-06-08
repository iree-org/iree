// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"
#include "iree/modules/hal/hal_module.h"
#include "iree/tools/utils/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/ref_cc.h"

IREE_FLAG(string, module_file, "-",
          "File containing the module to load that contains the entry "
          "function. Defaults to stdin.");

IREE_FLAG(string, entry_function, "",
          "Name of a function contained in the module specified by module_file "
          "to run.");

IREE_FLAG(string, driver, "vmvx", "Backend driver to use.");

static iree_status_t parse_function_input(iree_string_view_t flag_name,
                                          void* storage,
                                          iree_string_view_t value) {
  auto* list = (std::vector<std::string>*)storage;
  list->push_back(std::string(value.data, value.size));
  return iree_ok_status();
}
static void print_function_input(iree_string_view_t flag_name, void* storage,
                                 FILE* file) {
  auto* list = (std::vector<std::string>*)storage;
  if (list->empty()) {
    fprintf(file, "# --%.*s=\n", (int)flag_name.size, flag_name.data);
  } else {
    for (size_t i = 0; i < list->size(); ++i) {
      fprintf(file, "--%.*s=\"%s\"\n", (int)flag_name.size, flag_name.data,
              list->at(i).c_str());
    }
  }
}
static std::vector<std::string> FLAG_function_inputs;
IREE_FLAG_CALLBACK(
    parse_function_input, print_function_input, &FLAG_function_inputs,
    function_input,
    "An input value or buffer of the format:\n"
    "  [shape]xtype=[value]\n"
    "  2x2xi32=1 2 3 4\n"
    "Optionally, brackets may be used to separate the element values:\n"
    "  2x2xi32=[[1 2][3 4]]\n"
    "Each occurrence of the flag indicates an input in the order they were\n"
    "specified on the command line.");

namespace iree {
namespace {

iree_status_t GetModuleContentsFromFlags(std::string* out_contents) {
  IREE_TRACE_SCOPE0("GetModuleContentsFromFlags");
  auto module_file = std::string(FLAG_module_file);
  if (module_file == "-") {
    *out_contents = std::string{std::istreambuf_iterator<char>(std::cin),
                                std::istreambuf_iterator<char>()};
  } else {
    IREE_RETURN_IF_ERROR(GetFileContents(module_file.c_str(), out_contents));
  }
  return iree_ok_status();
}

iree_status_t Run() {
  IREE_TRACE_SCOPE0("iree-run-module");

  IREE_RETURN_IF_ERROR(iree_hal_module_register_types(),
                       "registering HAL types");
  iree_vm_instance_t* instance = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance),
      "creating instance");

  std::string module_data;
  IREE_RETURN_IF_ERROR(GetModuleContentsFromFlags(&module_data));
  iree_vm_module_t* input_module = nullptr;
  IREE_RETURN_IF_ERROR(LoadBytecodeModule(module_data, &input_module));

  iree_hal_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(CreateDevice(FLAG_driver, &device));
  iree_vm_module_t* hal_module = nullptr;
  IREE_RETURN_IF_ERROR(CreateHalModule(device, &hal_module));

  iree_vm_context_t* context = nullptr;
  // Order matters. The input module will likely be dependent on the hal module.
  std::array<iree_vm_module_t*, 2> modules = {hal_module, input_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
                           instance, modules.data(), modules.size(),
                           iree_allocator_system(), &context),
                       "creating context");

  std::string function_name = std::string(FLAG_entry_function);
  iree_vm_function_t function;
  if (function_name.empty()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no --entry_function= specified");
  } else {
    IREE_RETURN_IF_ERROR(
        input_module->lookup_function(
            input_module->self, IREE_VM_FUNCTION_LINKAGE_EXPORT,
            iree_string_view_t{function_name.data(), function_name.size()},
            &function),
        "looking up function '%s'", function_name.c_str());
  }

  vm::ref<iree_vm_list_t> inputs;
  IREE_CHECK_OK(ParseToVariantList(iree_hal_device_allocator(device),
                                   FLAG_function_inputs, &inputs));

  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(/*element_type=*/nullptr, 16,
                                           iree_allocator_system(), &outputs));

  std::cout << "EXEC @" << function_name << "\n";
  IREE_RETURN_IF_ERROR(
      iree_vm_invoke(context, function, /*policy=*/nullptr, inputs.get(),
                     outputs.get(), iree_allocator_system()),
      "invoking function '%s'", function_name.c_str());

  IREE_RETURN_IF_ERROR(PrintVariantList(outputs.get()), "printing results");

  inputs.reset();
  outputs.reset();
  iree_vm_module_release(hal_module);
  iree_vm_module_release(input_module);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return iree_ok_status();
}

}  // namespace

extern "C" int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  IREE_CHECK_OK(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));
  IREE_CHECK_OK(Run());
  return 0;
}

}  // namespace iree
