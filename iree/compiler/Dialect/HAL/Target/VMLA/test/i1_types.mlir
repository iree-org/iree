// RUN: iree-opt -split-input-file -pass-pipeline='iree-hal-transformation-pipeline{serialize-executables=false link-executables=false},canonicalize' -iree-hal-target-backends=vmla %s | IreeFileCheck %s

// CHECK-LABEL: @i1_op_usage(%arg0: !hal.buffer) -> !hal.buffer
func @i1_op_usage(%arg0: tensor<4xi1>) -> tensor<4xi1> {
  %c4 = constant 4 : index
  // CHECK: %0 = iree.byte_buffer.constant : !iree.byte_buffer = dense<[1, 0, 1, 0]> : tensor<4xi8>
  %cst = constant dense<[true, false, true, false]> : tensor<4xi1>
  %0 = flow.ex.stream.fragment(%arg1 = %c4 : index, %arg2 = %arg0 : tensor<4xi1>, %arg3 = %cst : tensor<4xi1>) -> tensor<4xi1> {
    %1 = flow.dispatch @i1_op_usage_ex_dispatch_0::@i1_op_usage_ex_dispatch_0[%arg1 : index](%arg2, %arg3) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    flow.return %1 : tensor<4xi1>
  }
  return %0 : tensor<4xi1>
}

// CHECK: hal.executable @i1_op_usage_ex_dispatch_0
// CHECK: hal.executable.target @vmla
// CHECK: hal.executable.entry_point @i1_op_usage_ex_dispatch_0 attributes {
// CHECK-SAME:   interface = @legacy_io
// CHECK-SAME:   ordinal = 0 : i32
// CHECK-SAME:   signature = (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
flow.executable @i1_op_usage_ex_dispatch_0 attributes {sym_visibility = "private"} {
  flow.dispatch.entry @i1_op_usage_ex_dispatch_0
  // CHECK: vm.module @module
  module {
    // CHECK: vm.rodata {{.+}} dense<[0, 0, 1, 0]> : tensor<4xi8>
    // CHECK: vm.func @i1_op_usage_ex_dispatch_0
    func @i1_op_usage_ex_dispatch_0(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
      %0 = mhlo.and %arg0, %arg1 : tensor<4xi1>
      %cst = mhlo.constant dense<[false, false, true, false]> : tensor<4xi1>
      %1 = mhlo.and %0, %cst : tensor<4xi1>
      return %1 : tensor<4xi1>
    }
  }
}
