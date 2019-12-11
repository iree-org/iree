// Tests printing and parsing of executable/structural ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @dispatch_ex
flow.executable @dispatch_ex {
  // CHECK: module {
  module {
    // CHECK: @dispatch0
    func @dispatch0() {
      return
    }
  }
  // CHECK: flow.dispatch.entry @dispatch0
  flow.dispatch.entry @dispatch0
  // CHECK: flow.dispatch.entry @dispatch0 as("dispatch0_alias")
  flow.dispatch.entry @dispatch0 as("dispatch0_alias")
}

// -----

// CHECK-LABEL: @reduction_ex
flow.executable @reduction_ex {
  // CHECK: module {
  module {
    // CHECK: @entry
    func @entry(tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    // CHECK: @apply
    func @apply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
      %0 = xla_hlo.add %arg0, %arg1 : tensor<f32>
      return %0 : tensor<f32>
    }
  }
  // CHECK: flow.reduction.entry @entry
  // CHECK-SAME: apply(@apply)
  // CHECK-SAME: as("entry_alias")
  // CHECK-SAME: attributes {dimension = 1 : i32}
  flow.reduction.entry @entry apply(@apply) as("entry_alias") attributes {dimension = 1 : i32}
}
