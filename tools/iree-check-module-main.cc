// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdio>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/api.h"
#include "iree/modules/check/module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

IREE_FLAG(
    bool, expect_failure, false,
    "Whether running module is expected to fail. If set, failing "
    "statuses from function evaluation are logged and ignored and all "
    "evaluations succeeding is considered an error and will return a failure. "
    "Mostly useful for testing the binary doesn't crash for failing tests.");

namespace iree {
namespace {

class CheckModuleTest : public ::testing::Test {
 public:
  explicit CheckModuleTest(iree_vm_instance_t* instance,
                           const iree_tooling_module_list_t* module_list,
                           iree_vm_function_t function)
      : instance_(instance), function_(function) {
    iree_tooling_module_list_clone(module_list, &module_list_);
  }
  ~CheckModuleTest() { iree_tooling_module_list_reset(&module_list_); }

  void SetUp() override {
    IREE_CHECK_OK(iree_tooling_create_context_from_flags(
        instance_, module_list_.count, module_list_.values,
        /*default_device_uri=*/iree_string_view_empty(),
        iree_vm_instance_allocator(instance_), &context_, &device_,
        /*out_device_allocator=*/NULL));
  }

  void TearDown() override {
    iree_vm_context_release(context_);
    iree_hal_device_release(device_);
  }

  void TestBody() override {
    IREE_ASSERT_OK(iree_hal_begin_profiling_from_flags(device_));
    IREE_EXPECT_OK(iree_vm_invoke(context_, function_,
                                  IREE_VM_INVOCATION_FLAG_NONE,
                                  /*policy=*/nullptr,
                                  /*inputs=*/nullptr, /*outputs=*/nullptr,
                                  iree_vm_instance_allocator(instance_)));
    IREE_ASSERT_OK(iree_hal_end_profiling_from_flags(device_));
  }

 private:
  iree_vm_instance_t* instance_ = nullptr;
  iree_tooling_module_list_t module_list_;
  iree_vm_function_t function_;

  iree_vm_context_t* context_ = nullptr;
  iree_hal_device_t* device_ = nullptr;
};

iree_status_t Run(iree_allocator_t host_allocator, int* out_exit_code) {
  *out_exit_code = 1;

  iree_vm_instance_t* instance = nullptr;
  IREE_RETURN_IF_ERROR(iree_tooling_create_instance(host_allocator, &instance),
                       "creating instance");

  iree_vm_module_t* check_module = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_check_module_create(instance, host_allocator, &check_module));

  // Resolve all system modules required by the user and check modules.
  iree_tooling_module_list_t module_list;
  iree_tooling_module_list_initialize(&module_list);
  IREE_RETURN_IF_ERROR(
      iree_tooling_module_list_push_back(&module_list, check_module));
  IREE_RETURN_IF_ERROR(iree_tooling_load_modules_from_flags(
      instance, host_allocator, &module_list));
  iree_vm_module_t* main_module = iree_tooling_module_list_back(&module_list);

  auto module_signature = iree_vm_module_signature(main_module);
  for (iree_host_size_t ordinal = 0;
       ordinal < module_signature.export_function_count; ++ordinal) {
    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(
        iree_vm_module_lookup_function_by_ordinal(
            main_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, ordinal, &function),
        "looking up function export %" PRIhsz, ordinal);
    iree_string_view_t function_name = iree_vm_function_name(&function);

    if (iree_string_view_starts_with(function_name,
                                     iree_make_cstring_view("__")) ||
        iree_string_view_find_char(function_name, '$', 0) !=
            IREE_STRING_VIEW_NPOS) {
      // Skip internal or special functions.
      continue;
    }

    iree_vm_function_signature_t signature =
        iree_vm_function_signature(&function);
    iree_host_size_t argument_count = 0;
    iree_host_size_t result_count = 0;
    IREE_RETURN_IF_ERROR(iree_vm_function_call_count_arguments_and_results(
        &signature, &argument_count, &result_count));
    if (argument_count || result_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected function with no inputs or outputs, "
                              "but export '%.*s' has signature '%.*s'",
                              (int)function_name.size, function_name.data,
                              (int)signature.calling_convention.size,
                              signature.calling_convention.data);
    }

    iree_string_view_t module_name = iree_vm_module_name(main_module);
    ::testing::RegisterTest(module_name.data, function_name.data, nullptr,
                            std::to_string(ordinal).c_str(), __FILE__, __LINE__,
                            [=]() -> CheckModuleTest* {
                              return new CheckModuleTest(instance, &module_list,
                                                         function);
                            });
  }
  *out_exit_code = RUN_ALL_TESTS();

  iree_tooling_module_list_reset(&module_list);
  iree_vm_module_release(check_module);
  iree_vm_instance_release(instance);

  return iree_ok_status();
}

}  // namespace

extern "C" int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();

  // Pass through flags to gtest (allowing --help to fall through).
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_UNDEFINED_OK |
                               IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                           &argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);

  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree-check-module");
  int exit_code = 1;
  iree_status_t status = Run(iree_allocator_system(), &exit_code);
  exit_code = iree_status_is_ok(status) ? exit_code : EXIT_FAILURE;
  IREE_TRACE_ZONE_END(z0);

  IREE_TRACE_APP_EXIT(exit_code);

  if (FLAG_expect_failure) {
    if (exit_code == 0) {
      printf("Test passed but expected failure\n");
      return 1;
    }
    printf("Test failed as expected\n");
    return 0;
  }

  if (exit_code != 0) {
    printf("Test failed\n%s\n", Status(std::move(status)).ToString().c_str());
  }

  return exit_code;
}

}  // namespace iree
