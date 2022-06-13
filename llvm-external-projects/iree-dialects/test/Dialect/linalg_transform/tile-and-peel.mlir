// RUN: iree-dialects-opt --transform-dialect-interpreter %s | FileCheck %s

// CHECK-LABEL: func.func @matmul_tensors(
func.func @matmul_tensors(
  %arg0: tensor<126x127xf32>, %arg1: tensor<127x128xf32>, %arg2: tensor<126x128xf32> { linalg.inplaceable = true})
    -> tensor<126x128xf32> {
  // CHECK-DAG: %[[c124:.*]] = arith.constant 124 : index
  // CHECK-DAG: %[[c127:.*]] = arith.constant 127 : index
  // CHECK-DAG: %[[c128:.*]] = arith.constant 128 : index

  // CHECK: scf.for {{.*}} to %[[c124]]
  // CHECK:   scf.for {{.*}} to %[[c128]]
  // CHECK:     scf.for {{.*}} to %[[c124]]
  // CHECK:       linalg.matmul ins({{.*}} : tensor<4x4xf32>, tensor<4x4xf32>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK:     linalg.matmul ins({{.*}} : tensor<4x3xf32>, tensor<3x4xf32>) outs({{.*}} : tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: scf.for {{.*}} to %[[c128]]
  // CHECK:   scf.for {{.*}} to %[[c127]]
  // CHECK:     linalg.matmul ins({{.*}} : tensor<2x?xf32>, tensor<?x4xf32>) outs({{.*}} : tensor<2x4xf32>) -> tensor<2x4xf32>
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<126x127xf32>, tensor<127x128xf32>)
                     outs(%arg2: tensor<126x128xf32>)
    -> tensor<126x128xf32>

  return %0 : tensor<126x128xf32>
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
    %0 = pdl_match @pdl_target in %arg1
    %linalg_op, %loops:3 = transform.structured.tile %0 {sizes = [4, 4, 4]}

    // Note: The order in which the loops are peeled is important. If %loop#2 is
    // peeled first, the partial iteration of %loop#0 will also contain a peeled
    // version of %loop#2.
    transform.loop.peel %loops#0
    transform.loop.peel %loops#2
  }
}
