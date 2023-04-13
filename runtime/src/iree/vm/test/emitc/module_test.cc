// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: We should not be including C implementation-only headers in a C++
// module like this. In order to make this work for the moment across
// runtime libraries that are strict, do a global using of the std namespace.
// See #7605
#include <cmath>
using namespace std;

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/vm/api.h"
#define EMITC_IMPLEMENTATION
#include "iree/vm/test/emitc/arithmetic_ops.h"
#include "iree/vm/test/emitc/arithmetic_ops_f32.h"
#include "iree/vm/test/emitc/arithmetic_ops_i64.h"
#include "iree/vm/test/emitc/assignment_ops.h"
#include "iree/vm/test/emitc/assignment_ops_f32.h"
#include "iree/vm/test/emitc/assignment_ops_i64.h"
#include "iree/vm/test/emitc/buffer_ops.h"
#include "iree/vm/test/emitc/call_ops.h"
#include "iree/vm/test/emitc/comparison_ops.h"
#include "iree/vm/test/emitc/comparison_ops_f32.h"
#include "iree/vm/test/emitc/comparison_ops_i64.h"
#include "iree/vm/test/emitc/control_flow_ops.h"
#include "iree/vm/test/emitc/conversion_ops.h"
#include "iree/vm/test/emitc/conversion_ops_f32.h"
#include "iree/vm/test/emitc/conversion_ops_i64.h"
#include "iree/vm/test/emitc/global_ops.h"
#include "iree/vm/test/emitc/global_ops_f32.h"
#include "iree/vm/test/emitc/global_ops_i64.h"
#include "iree/vm/test/emitc/list_ops.h"
#include "iree/vm/test/emitc/list_variant_ops.h"
#include "iree/vm/test/emitc/ref_ops.h"
#include "iree/vm/test/emitc/shift_ops.h"
#include "iree/vm/test/emitc/shift_ops_i64.h"

namespace {

typedef iree_status_t (*create_function_t)(iree_vm_instance_t*,
                                           iree_allocator_t,
                                           iree_vm_module_t**);

struct TestParams {
  std::string module_name;
  std::string local_name;
  create_function_t create_function;
};

struct ModuleDescription {
  iree_vm_native_module_descriptor_t descriptor;
  create_function_t create_function;
};

std::ostream& operator<<(std::ostream& os, const TestParams& params) {
  std::string qualified_name = params.module_name + "." + params.local_name;
  auto name_sv =
      iree_make_string_view(qualified_name.data(), qualified_name.size());
  iree_string_view_replace_char(name_sv, ':', '_');
  iree_string_view_replace_char(name_sv, '.', '_');
  return os << qualified_name;
}

std::vector<TestParams> GetModuleTestParams() {
  std::vector<TestParams> test_params;

  // TODO(simon-camp): get these automatically
  std::vector<ModuleDescription> modules = {
      {arithmetic_ops_descriptor_, arithmetic_ops_create},
      {arithmetic_ops_f32_descriptor_, arithmetic_ops_f32_create},
      {arithmetic_ops_i64_descriptor_, arithmetic_ops_i64_create},
      {assignment_ops_descriptor_, assignment_ops_create},
      {assignment_ops_f32_descriptor_, assignment_ops_f32_create},
      {assignment_ops_i64_descriptor_, assignment_ops_i64_create},
      {buffer_ops_descriptor_, buffer_ops_create},
      {call_ops_descriptor_, call_ops_create},
      {comparison_ops_descriptor_, comparison_ops_create},
      {comparison_ops_f32_descriptor_, comparison_ops_f32_create},
      {comparison_ops_i64_descriptor_, comparison_ops_i64_create},
      {control_flow_ops_descriptor_, control_flow_ops_create},
      {conversion_ops_descriptor_, conversion_ops_create},
      {conversion_ops_f32_descriptor_, conversion_ops_f32_create},
      {conversion_ops_i64_descriptor_, conversion_ops_i64_create},
      {global_ops_descriptor_, global_ops_create},
      {global_ops_f32_descriptor_, global_ops_f32_create},
      {global_ops_i64_descriptor_, global_ops_i64_create},
      {list_ops_descriptor_, list_ops_create},
      {list_variant_ops_descriptor_, list_variant_ops_create},
      {ref_ops_descriptor_, ref_ops_create},
      {shift_ops_descriptor_, shift_ops_create},
      {shift_ops_i64_descriptor_, shift_ops_i64_create}};

  for (size_t i = 0; i < modules.size(); i++) {
    iree_vm_native_module_descriptor_t descriptor = modules[i].descriptor;
    create_function_t function = modules[i].create_function;

    std::string module_name =
        std::string(descriptor.name.data, descriptor.name.size);

    for (iree_host_size_t i = 0; i < descriptor.export_count; i++) {
      iree_vm_native_export_descriptor_t export_descriptor =
          descriptor.exports[i];
      std::string local_name = std::string(export_descriptor.local_name.data,
                                           export_descriptor.local_name.size);
      test_params.push_back({module_name, local_name, function});
    }
  }

  return test_params;
}

class VMCModuleTest : public ::testing::Test,
                      public ::testing::WithParamInterface<TestParams> {
 protected:
  virtual void SetUp() {
    const auto& test_params = GetParam();

    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          iree_allocator_system(), &instance_));

    iree_vm_module_t* module_ = nullptr;
    IREE_CHECK_OK(test_params.create_function(
        instance_, iree_allocator_system(), &module_));

    std::vector<iree_vm_module_t*> modules = {module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));

    iree_vm_module_release(module_);
  }

  virtual void TearDown() {
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  iree_status_t RunFunction(std::string module_name, std::string local_name) {
    std::string qualified_name = module_name + "." + local_name;
    iree_vm_function_t function;
    IREE_CHECK_OK(iree_vm_context_resolve_function(
        context_,
        iree_string_view_t{qualified_name.data(), qualified_name.size()},
        &function));

    return iree_vm_invoke(context_, function, IREE_VM_INVOCATION_FLAG_NONE,
                          /*policy=*/nullptr, /*inputs=*/nullptr,
                          /*outputs=*/nullptr, iree_allocator_system());
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
};

TEST_P(VMCModuleTest, Check) {
  const auto& test_params = GetParam();
  bool expect_failure = test_params.local_name.find("fail_") == 0;

  iree::Status result =
      RunFunction(test_params.module_name, test_params.local_name);
  if (result.ok()) {
    if (expect_failure) {
      GTEST_FAIL() << "Function expected failure but succeeded";
    } else {
      GTEST_SUCCEED();
    }
  } else {
    if (expect_failure) {
      GTEST_SUCCEED();
    } else {
      GTEST_FAIL() << "Function expected success but failed with error: "
                   << result.ToString();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(VMIRFunctions, VMCModuleTest,
                         ::testing::ValuesIn(GetModuleTestParams()),
                         ::testing::PrintToStringParamName());

}  // namespace
