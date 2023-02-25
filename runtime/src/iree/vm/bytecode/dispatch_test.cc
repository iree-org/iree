// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests covering the dispatch logic for individual ops.
//
// iree/vm/test/*.mlir contains the functions used here for testing. We
// avoid defining the IR inline here so that we can run this test on platforms
// that we can't run the full MLIR compiler stack on.

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// Compiled module embedded here to avoid file IO:
#include "iree/vm/test/all_bytecode_modules.h"

namespace {

struct TestParams {
  const struct iree_file_toc_t& module_file;
  std::string function_name;
};

std::ostream& operator<<(std::ostream& os, const TestParams& params) {
  std::string name{params.module_file.name};
  auto name_sv = iree_make_string_view(name.data(), name.size());
  iree_string_view_replace_char(name_sv, ':', '_');
  iree_string_view_replace_char(name_sv, '.', '_');
  return os << name << "_" << params.function_name;
}

std::vector<TestParams> GetModuleTestParams() {
  std::vector<TestParams> test_params;

  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance));

  const struct iree_file_toc_t* module_file_toc =
      all_bytecode_modules_c_create();
  for (size_t i = 0; i < all_bytecode_modules_c_size(); ++i) {
    const auto& module_file = module_file_toc[i];
    iree_vm_module_t* module = nullptr;
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance,
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file.data),
            module_file.size},
        iree_allocator_null(), iree_allocator_system(), &module));
    iree_vm_module_signature_t signature = iree_vm_module_signature(module);
    test_params.reserve(test_params.size() + signature.export_function_count);
    for (int i = 0; i < signature.export_function_count; ++i) {
      iree_vm_function_t function;
      IREE_CHECK_OK(iree_vm_module_lookup_function_by_ordinal(
          module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &function));
      iree_string_view_t function_name = iree_vm_function_name(&function);
      test_params.push_back(
          {module_file, std::string(function_name.data, function_name.size)});
    }
    iree_vm_module_release(module);
  }

  iree_vm_instance_release(instance);

  return test_params;
}

class VMBytecodeDispatchTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<TestParams> {
 protected:
  virtual void SetUp() {
    const auto& test_params = GetParam();

    IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance_));

    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance_,
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(test_params.module_file.data),
            test_params.module_file.size},
        iree_allocator_null(), iree_allocator_system(), &bytecode_module_));

    std::vector<iree_vm_module_t*> modules = {bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));
  }

  virtual void TearDown() {
    iree_vm_module_release(bytecode_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  iree_status_t RunFunction(const char* function_name) {
    iree_vm_function_t function;
    IREE_CHECK_OK(iree_vm_module_lookup_function_by_name(
        bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_make_cstring_view(function_name), &function));

    return iree_vm_invoke(context_, function, IREE_VM_INVOCATION_FLAG_NONE,
                          /*policy=*/nullptr, /*inputs=*/nullptr,
                          /*outputs=*/nullptr, iree_allocator_system());
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
};

TEST_P(VMBytecodeDispatchTest, Check) {
  const auto& test_params = GetParam();
  bool expect_failure = test_params.function_name.find("fail_") == 0;

  iree_status_t status = RunFunction(test_params.function_name.c_str());
  if (iree_status_is_ok(status)) {
    if (expect_failure) {
      GTEST_FAIL() << "Function expected failure but succeeded";
    } else {
      GTEST_SUCCEED();
    }
  } else {
    if (expect_failure) {
      iree_status_ignore(status);
      GTEST_SUCCEED();
    } else {
      GTEST_FAIL() << "Function expected success but failed with error: "
                   << iree::Status(std::move(status)).ToString();
    }
  }
}

INSTANTIATE_TEST_SUITE_P(VMIRFunctions, VMBytecodeDispatchTest,
                         ::testing::ValuesIn(GetModuleTestParams()),
                         ::testing::PrintToStringParamName());

}  // namespace
