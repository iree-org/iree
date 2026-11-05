// RUN: iree-opt --split-input-file --iree-dispatch-creation-assign-data-tiling-encodings %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-dispatch-creation-assign-data-tiling-encodings{data-tiling-op-types=convolution})' %s | FileCheck %s --check-prefix=CONV-ONLY

// CHECK-LABEL: util.func public @assign_encodings(
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<64x128xf32>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<128x64xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9]+]]: tensor<64x64xf32>
// CHECK:         %[[ENCODED_LHS:.+]] = iree_encoding.set_encoding %[[LHS]]
// CHECK:         %[[ENCODED_RHS:.+]] = iree_encoding.set_encoding %[[RHS]]
// CHECK:         %[[ENCODED_INIT:.+]] = iree_encoding.set_encoding %[[INIT]]
// CHECK:         %[[MATMUL:.+]] = linalg.matmul ins(%[[ENCODED_LHS]], %[[ENCODED_RHS]]
// CHECK-SAME:      outs(%[[ENCODED_INIT]]
// CHECK:         %[[RESULT:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
// CHECK:         util.return %[[RESULT]]
// With only convolutions selected, the matmul keeps its plain tensors.
// CONV-ONLY-LABEL: util.func public @assign_encodings(
// CONV-ONLY-NOT:     iree_encoding
// CONV-ONLY:         linalg.matmul
// CONV-ONLY-NOT:     iree_encoding
// CONV-ONLY:         util.return
util.func public @assign_encodings(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
  util.return %result : tensor<64x64xf32>
}

// -----

// Layouts were already materialized, so encodings are not assigned again.
#target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf"}>
module attributes {iree.encoding.materialized_layout_target = #target} {
  // CHECK-LABEL: util.func public @no_assign_materialized_layouts(
  // CHECK-NOT:     iree_encoding
  // CHECK:         linalg.matmul ins(%arg0, %arg1
  // CHECK-NOT:     iree_encoding
  // CHECK:         util.return
  util.func public @no_assign_materialized_layouts(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>, %init: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %result = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    util.return %result : tensor<64x64xf32>
  }
}
