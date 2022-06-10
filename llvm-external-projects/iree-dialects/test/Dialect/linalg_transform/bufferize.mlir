// RUN: iree-dialects-opt --linalg-transform-interp %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_tensors(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: memref<128x128xf32
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: memref<128x128xf32
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: memref<128x128xf32
// CHECK-NOT:   -> tensor
func.func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // CHECK: linalg.matmul ins(%[[TA]], %[[TB]] : memref{{.*}}, memref{{.*}} outs(%[[TC]] : memref{{.*}})
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  // CHECK: return %[[TC]]
  return %0 : tensor<128x128xf32>
// CHECK: }
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_target : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    %1 = pdl.attribute = @matmul_tensors
    apply_native_constraint "nestedInFunc"(%0, %1 : !pdl.operation, !pdl.attribute)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    bufferize
  }
}
