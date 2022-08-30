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
#include "iree/base/internal/flags.h"
#include "iree/base/status_cc.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/vm_util.h"
#include "iree/vm/api.h"
#include "iree/vm/ref_cc.h"

IREE_FLAG(string, entry_function, "",
          "Name of a function contained in the module specified by module_file "
          "to run.");

IREE_FLAG(int32_t, print_max_element_count, 1024,
          "Prints up to the maximum number of elements of output tensors, "
          "eliding the remainder.");

IREE_FLAG(bool, print_statistics, false,
          "Prints runtime statistics to stderr on exit.");

// TODO(benvanik): move --function_input= flag into a util.
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
    "Raw binary files can be read to provide buffer contents:\n"
    "  2x2xi32=@some/file.bin\n"
    "numpy npy files (from numpy.save) can be read to provide 1+ values:\n"
    "  @some.npy\n"
    "Each occurrence of the flag indicates an input in the order they were\n"
    "specified on the command line.");

namespace iree {
namespace {

iree_status_t Run() {
  IREE_TRACE_SCOPE0("iree-run-module");

  iree_allocator_t host_allocator = iree_allocator_system();
  iree_vm_instance_t* instance = nullptr;
  IREE_RETURN_IF_ERROR(iree_tooling_create_instance(host_allocator, &instance),
                       "creating instance");

  iree_vm_module_t* main_module = nullptr;
  IREE_RETURN_IF_ERROR(iree_tooling_load_module_from_flags(
      instance, host_allocator, &main_module));

  iree_vm_context_t* context = NULL;
  iree_hal_device_t* device = NULL;
  iree_hal_allocator_t* device_allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_tooling_create_context_from_flags(
      instance, /*user_module_count=*/1, /*user_modules=*/&main_module,
      /*default_device_uri=*/iree_string_view_empty(), host_allocator, &context,
      &device, &device_allocator));

  std::string function_name = std::string(FLAG_entry_function);
  iree_vm_function_t function;
  if (function_name.empty()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no --entry_function= specified");
  } else {
    IREE_RETURN_IF_ERROR(
        iree_vm_module_lookup_function_by_name(
            main_module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
            iree_string_view_t{function_name.data(), function_name.size()},
            &function),
        "looking up function '%s'", function_name.c_str());
  }

  vm::ref<iree_vm_list_t> inputs;
  IREE_RETURN_IF_ERROR(ParseToVariantList(
      device_allocator,
      iree::span<const std::string>{FLAG_function_inputs.data(),
                                    FLAG_function_inputs.size()},
      host_allocator, &inputs));

  vm::ref<iree_vm_list_t> outputs;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(/*element_type=*/nullptr, 16,
                                           host_allocator, &outputs));

  std::cout << "EXEC @" << function_name << "\n";
  IREE_RETURN_IF_ERROR(
      iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
                     /*policy=*/nullptr, inputs.get(), outputs.get(),
                     host_allocator),
      "invoking function '%s'", function_name.c_str());

  IREE_RETURN_IF_ERROR(
      PrintVariantList(outputs.get(), (size_t)FLAG_print_max_element_count),
      "printing results");

  // Release resources before gathering statistics.
  inputs.reset();
  outputs.reset();
  iree_vm_module_release(main_module);
  iree_vm_context_release(context);

  if (FLAG_print_statistics) {
    IREE_IGNORE_ERROR(
        iree_hal_allocator_statistics_fprint(stderr, device_allocator));
  }

  iree_hal_allocator_release(device_allocator);
  iree_hal_device_release(device);
  iree_vm_instance_release(instance);
  return iree_ok_status();
}

}  // namespace

extern "C" int main(int argc, char** argv) {
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);
  if (argc > 1) {
    // Avoid iree-run-module spinning endlessly on stdin if the user uses single
    // dashes for flags.
    std::cout << "Error: unexpected positional argument (expected none)."
                 " Did you use pass a flag with a single dash ('-')?"
                 " Use '--' instead.\n";
    return 1;
  }
  IREE_CHECK_OK(Run());
  return 0;
}

}  // namespace iree
