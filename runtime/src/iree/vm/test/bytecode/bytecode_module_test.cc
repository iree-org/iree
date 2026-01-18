// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>
#include <string>

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "iree/vm/test/all_bytecode_modules.h"
#include "iree/vm/testing/test_runner.h"
#include "iree/vm/testing/yieldable_test_module.h"

IREE_VM_TEST_RUNNER_STATIC_STORAGE();

namespace iree::vm::testing {
namespace {

std::vector<VMTestParams> GetBytecodeTestParams() {
  std::vector<VMTestParams> test_params;

  // Prerequisite factory for modules that import from yieldable_test.
  auto yieldable_test_factory = [](iree_vm_instance_t* inst,
                                   iree_vm_module_t** out_mod) {
    return yieldable_test_module_create(inst, iree_allocator_system(), out_mod);
  };

  iree_vm_instance_t* instance = nullptr;
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        iree_allocator_system(), &instance));

  const struct iree_file_toc_t* module_file_toc =
      all_bytecode_modules_c_create();
  for (size_t i = 0; i < all_bytecode_modules_c_size(); ++i) {
    const auto& module_file = module_file_toc[i];
    std::string module_name(module_file.name);

    iree_vm_module_t* module = nullptr;
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file.data),
            static_cast<iree_host_size_t>(module_file.size)},
        iree_allocator_null(), iree_allocator_system(), &module));

    iree_vm_module_signature_t signature = iree_vm_module_signature(module);
    for (iree_host_size_t j = 0; j < signature.export_function_count; ++j) {
      iree_vm_function_t function;
      IREE_CHECK_OK(iree_vm_module_lookup_function_by_ordinal(
          module, IREE_VM_FUNCTION_LINKAGE_EXPORT, j, &function));
      iree_string_view_t function_name = iree_vm_function_name(&function);
      std::string fn_name(function_name.data, function_name.size);

      // Capture module data for lambda.
      const void* data = module_file.data;
      iree_host_size_t size = module_file.size;

      std::vector<VMModuleCreateFn> prereqs;
      prereqs.push_back(yieldable_test_factory);

      test_params.push_back({
          module_name,
          fn_name,
          [data, size](iree_vm_instance_t* inst, iree_vm_module_t** out_mod) {
            return iree_vm_bytecode_module_create(
                inst, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
                iree_const_byte_span_t{reinterpret_cast<const uint8_t*>(data),
                                       static_cast<iree_host_size_t>(size)},
                iree_allocator_null(), iree_allocator_system(), out_mod);
          },
          /*expects_failure=*/fn_name.find("fail_") == 0,
          /*prerequisite_modules=*/prereqs,
      });
    }
    iree_vm_module_release(module);
  }

  iree_vm_instance_release(instance);
  return test_params;
}

class VMBytecodeTest : public VMTestRunner<> {};

IREE_VM_TEST_F(VMBytecodeTest)

INSTANTIATE_TEST_SUITE_P(bytecode, VMBytecodeTest,
                         ::testing::ValuesIn(GetBytecodeTestParams()),
                         ::testing::PrintToStringParamName());

}  // namespace
}  // namespace iree::vm::testing
