// RUN: iree-opt %s | iree-opt | FileCheck %s

module @tuning_module attributes {
  rocm.spec = #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir"> } {
}

// CHECK-LABEL: module @tuning_module
//  CHECK-SAME:   #rocm.builtin.tuning_module<"iree_default_tuning_spec_gfx942.mlir">
