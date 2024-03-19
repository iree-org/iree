// RUN: iree-opt --iree-global-opt-optimize-numerics %s | FileCheck %s

// CHECK-LABEL: @matmul_i8_i8_i32_unsigned
util.func public @matmul_i8_i8_i32_unsigned(%arg0 : tensor<5x3xf32>, %arg1 : tensor<3x1xf32>, %arg2 : tensor<5x1xf32>) -> tensor<5x1xf32> {
  // CHECK: %[[LHS:.*]] = arith.fptoui %arg0 : tensor<5x3xf32> to tensor<5x3xi8>
  // CHECK: %[[RHS:.*]] = arith.fptoui %arg1 : tensor<3x1xf32> to tensor<3x1xi8>
  // CHECK: %[[INIT:.*]] = arith.fptoui %arg2 : tensor<5x1xf32> to tensor<5x1xi32>
  %lhs = util.numeric.optional_narrow %arg0 : tensor<5x3xf32> as ui7 {max_value = 127 : ui7, min_value = 0 : ui7}
  %rhs = util.numeric.optional_narrow %arg1 : tensor<3x1xf32> as ui7 {max_value = 127 : ui7, min_value = 0 : ui7}
  %init = util.numeric.optional_narrow %arg2 : tensor<5x1xf32> as ui0
  // CHECK: %[[RESULT:.*]] = linalg.matmul_unsigned ins(%[[LHS]], %[[RHS]] : tensor<5x3xi8>, tensor<3x1xi8>) outs(%[[INIT]] : tensor<5x1xi32>)
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<5x3xf32>, tensor<3x1xf32>) outs(%init : tensor<5x1xf32>) -> tensor<5x1xf32>
  // CHECK: arith.uitofp %[[RESULT]] : tensor<5x1xi32> to tensor<5x1xf32>
  util.return %2 : tensor<5x1xf32>
}

// CHECK-LABEL: @matmul_i8_i8_i32_signed
util.func public @matmul_i8_i8_i32_signed(%arg0 : tensor<5x3xf32>, %arg1 : tensor<3x1xf32>, %arg2 : tensor<5x1xf32>) -> tensor<5x1xf32> {
  // CHECK: %[[LHS:.*]] = arith.fptosi %arg0 : tensor<5x3xf32> to tensor<5x3xi8>
  // CHECK: %[[RHS:.*]] = arith.fptosi %arg1 : tensor<3x1xf32> to tensor<3x1xi8>
  // CHECK: %[[INIT:.*]] = arith.fptosi %arg2 : tensor<5x1xf32> to tensor<5x1xi32>
  %lhs = util.numeric.optional_narrow %arg0 : tensor<5x3xf32> as ui7 {max_value = 127 : ui7, min_value = 0 : ui7}
  %rhs = util.numeric.optional_narrow %arg1 : tensor<3x1xf32> as si8 {max_value = 127 : si8, min_value = -127 : si8}
  %init = util.numeric.optional_narrow %arg2 : tensor<5x1xf32> as ui0
  // CHECK: %[[RESULT:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<5x3xi8>, tensor<3x1xi8>) outs(%[[INIT]] : tensor<5x1xi32>)
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<5x3xf32>, tensor<3x1xf32>) outs(%init : tensor<5x1xf32>) -> tensor<5x1xf32>
  // CHECK: arith.sitofp %[[RESULT]] : tensor<5x1xi32> to tensor<5x1xf32>
  util.return %2 : tensor<5x1xf32>
}

// CHECK-LABEL: @matmul_i4_i4_i32_signed
// For now we clamp this to i8
util.func public @matmul_i4_i4_i32_signed(%arg0 : tensor<5x3xf32>, %arg1 : tensor<3x1xf32>, %arg2 : tensor<5x1xf32>) -> tensor<5x1xf32> {
  // CHECK: %[[LHS:.*]] = arith.fptosi %arg0 : tensor<5x3xf32> to tensor<5x3xi8>
  // CHECK: %[[RHS:.*]] = arith.fptosi %arg1 : tensor<3x1xf32> to tensor<3x1xi8>
  // CHECK: %[[INIT:.*]] = arith.fptosi %arg2 : tensor<5x1xf32> to tensor<5x1xi32>
  %lhs = util.numeric.optional_narrow %arg0 : tensor<5x3xf32> as si4 {max_value = 7 : si4, min_value = -7 : si4}
  %rhs = util.numeric.optional_narrow %arg1 : tensor<3x1xf32> as si4 {max_value = 3 : si4, min_value = -7 : si4}
  %init = util.numeric.optional_narrow %arg2 : tensor<5x1xf32> as ui0
  // CHECK: %[[RESULT:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<5x3xi8>, tensor<3x1xi8>) outs(%[[INIT]] : tensor<5x1xi32>)
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<5x3xf32>, tensor<3x1xf32>) outs(%init : tensor<5x1xf32>) -> tensor<5x1xf32>
  // CHECK: arith.sitofp %[[RESULT]] : tensor<5x1xi32> to tensor<5x1xf32>
  util.return %2 : tensor<5x1xf32>
}

// CHECK-LABEL: @matmul_reject_gt_8bit
// We may relax this restriction at some point but for right now we have it
// because less analysis is needed to prove safety.
// CHECK-NOT: fptosi
util.func public @matmul_reject_gt_8bit(%arg0 : tensor<5x3xf32>, %arg1 : tensor<3x1xf32>, %arg2 : tensor<5x1xf32>) -> tensor<5x1xf32> {
  %lhs = util.numeric.optional_narrow %arg0 : tensor<5x3xf32> as ui9 {max_value = 312 : ui9, min_value = 0 : ui9}
  %rhs = util.numeric.optional_narrow %arg1 : tensor<3x1xf32> as si8 {max_value = 127 : si8, min_value = -127 : si8}
  %init = util.numeric.optional_narrow %arg2 : tensor<5x1xf32> as ui0
  // CHECK: linalg.matmul {{.*}} -> tensor<5x1xf32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<5x3xf32>, tensor<3x1xf32>) outs(%init : tensor<5x1xf32>) -> tensor<5x1xf32>
  util.return %2 : tensor<5x1xf32>
}

// CHECK-LABEL: @cast_fill
util.func public @cast_fill(%arg0 : f32, %arg1 : tensor<3xf32>) -> tensor<3xi8> {
  // CHECK: %[[SCALAR:.*]] = arith.fptosi %arg0 : f32 to i8
  // CHECK: %[[INIT:.*]] = arith.fptosi %arg1 : tensor<3xf32> to tensor<3xi8>
  // CHECK: %[[RESULT:.*]] = linalg.fill ins(%[[SCALAR]] : i8) outs(%[[INIT]] : tensor<3xi8>) -> tensor<3xi8>
  // CHECK: util.return %[[RESULT]]
  %0 = linalg.fill ins(%arg0 : f32) outs(%arg1 : tensor<3xf32>) -> tensor<3xf32>
  %1 = arith.fptosi %0 : tensor<3xf32> to tensor<3xi8>
  util.return %1 : tensor<3xi8>
}

// CHECK-LABEL: @cast_init
util.func public @cast_init() -> tensor<5x9xi8> {
  // CHECK: %[[RESULT:.*]] = tensor.empty() : tensor<5x9xi8>
  // CHECK: util.return %[[RESULT]]
  %0 = tensor.empty() : tensor<5x9xf32>
  %1 = arith.fptosi %0 : tensor<5x9xf32> to tensor<5x9xi8>
  util.return %1 : tensor<5x9xi8>
}
