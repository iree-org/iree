// RUN: iree-opt --split-input-file --iree-global-optimization-hoist-constant-expressions --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @hoist_sub_byte_tensor_store
module @hoist_sub_byte_tensor_store {
  func.func @main() -> (tensor<64xi4>) {
    %0 = arith.constant dense<3> : tensor<64xi32>
    %2 = "iree_unregistered.const_expr"(%0) : (tensor<64xi32>) -> tensor<64xi4>
    return %2 : tensor<64xi4>
  }
}

// CHECK: util.global private @{{.*}} : tensor<32xi8>
// CHECK: func.func @main() -> tensor<64xi4>
// CHECK:   %[[GLOBAL_LD:.+]] = util.global.load @{{.*}} : tensor<32xi8>
// CHECK:   %[[ORIG_VAL:.+]] = flow.tensor.bitcast %[[GLOBAL_LD]] : tensor<32xi8> -> tensor<64xi4>
// CHECK:   return %[[ORIG_VAL]]

// CHECK: util.initializer attributes {iree.compiler.consteval}
// CHECK:   %[[CEXPR:.+]] = "iree_unregistered.const_expr"
// CHECK:   %[[CASTED_GLOBAL:.+]] = flow.tensor.bitcast %[[CEXPR]] : tensor<64xi4> -> tensor<32xi8>
// CHECK:   util.global.store %[[CASTED_GLOBAL]]
// CHECK:   util.initializer.return

// -----

// CHECK-LABEL: @hoist_sub_byte_tensor_transitive
// CHECK: util.global
module @hoist_sub_byte_tensor_transitive {
  func.func @main() -> (i32) {
    %0 = arith.constant dense<3> : tensor<i4>
    %2 = "iree_unregistered.const_expr"(%0) : (tensor<i4>) -> i32
    return %2 : i32
  }
}
// We do not need to cast for transitive sub-byte values.
// CHECK-NOT: flow.tensor.bitcast
