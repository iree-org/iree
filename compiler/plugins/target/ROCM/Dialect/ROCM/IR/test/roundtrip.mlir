// RUN: iree-opt %s | iree-opt | FileCheck %s

module @tuning_module attributes {
  iree_codegen.ukernel_provider = #rocm.ukernel_provider,
  rocm.spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir"> } {
}

// CHECK-LABEL: module @tuning_module
//  CHECK-SAME:   iree_codegen.ukernel_provider = #rocm.ukernel_provider
//  CHECK-SAME:   #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">
