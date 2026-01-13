// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

// We should not be including C implementation-only headers in a C++
// module like this. In order to make this work for the moment across
// runtime libraries that are strict, do a global using of the std namespace.
// EmitC is deprecated and will not be gaining any additional test support so
// this is an "as long as it works it's fine" compromise.
using namespace std;

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "iree/vm/testing/test_runner.h"

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

IREE_VM_TEST_RUNNER_STATIC_STORAGE();

namespace iree::vm::testing {
namespace {

typedef iree_status_t (*emitc_create_fn_t)(iree_vm_instance_t*,
                                           iree_allocator_t,
                                           iree_vm_module_t**);

struct EmitcModuleInfo {
  iree_vm_native_module_descriptor_t descriptor;
  emitc_create_fn_t create_fn;
};

std::vector<VMTestParams> GetEmitcTestParams() {
  std::vector<VMTestParams> test_params;

  std::vector<EmitcModuleInfo> modules = {
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
      {shift_ops_i64_descriptor_, shift_ops_i64_create},
  };

  for (const auto& mod : modules) {
    std::string module_name(mod.descriptor.name.data, mod.descriptor.name.size);
    emitc_create_fn_t create_fn = mod.create_fn;

    for (iree_host_size_t i = 0; i < mod.descriptor.export_count; ++i) {
      const iree_vm_native_export_descriptor_t& export_desc =
          mod.descriptor.exports[i];
      std::string fn_name(export_desc.local_name.data,
                          export_desc.local_name.size);
      test_params.push_back({
          module_name,
          fn_name,
          [create_fn](iree_vm_instance_t* inst, iree_vm_module_t** out_mod) {
            return create_fn(inst, iree_allocator_system(), out_mod);
          },
          /*expects_failure=*/fn_name.find("fail_") == 0,
          /*prerequisite_modules=*/{},
      });
    }
  }

  return test_params;
}

class VMEmitcTest : public VMTestRunner<> {};

IREE_VM_TEST_F(VMEmitcTest)

INSTANTIATE_TEST_SUITE_P(emitc, VMEmitcTest,
                         ::testing::ValuesIn(GetEmitcTestParams()),
                         ::testing::PrintToStringParamName());

}  // namespace
}  // namespace iree::vm::testing
