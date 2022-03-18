// RUN: iree-dialects-opt %s -linalg-interp-transforms -split-input-file | FileCheck %s

// CHECK-LABEL: func @matmul_tensors(
func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>,
  %arg3: tensor<128x128xf32>, %arg4: tensor<128x128xf32>, %arg5: tensor<128x128xf32>,
  %arg6: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // This operation is marked for tiling only.
  // CHECK-COUNT-3: scf.for
  // CHECK-COUNT-3: tensor.extract_slice
  // CHECK: linalg.matmul
  // CHECK-SAME: -> tensor<4x4xf32>
  %0 = linalg.matmul { test.attrA}
                      ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  // This operation is marked for tiling and vectorization.
  // Note that the loop-invariant read is hoisted out of the innermost loop.
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     vector.transfer_read
  // CHECK:     scf.for
  // CHECK:       vector.transfer_read
  // CHECK:       vector.transfer_read
  // CHECK:       vector.contract
  // CHECK-NOT:   linalg.matmul
  // CHECK:       vector.transfer_write
  %1 = linalg.matmul { test.attrA, test.attrC}
                      ins(%arg3, %arg4: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg5: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  // This operation is marked for vectorization only.
  // CHECK-NOT: scf.for
  // CHECK-COUNT-3: vector.transfer_read
  // CHECK: vector.contract
  // CHECK-SAME: into vector<128x128xf32>
  // CHECK: vector.transfer_write
  %2 = linalg.matmul { test.attrC}
                      ins(%0, %1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg6: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %2 : tensor<128x128xf32>
}

// Match matmul operations inside @matmul_tensors with test.attrA set.
pdl.pattern @pdl_target_attrA : benefit(1) {
  %args = operands
  %results = types
  %attr = attribute
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrA" = %attr}-> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@matmul_tensors](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

// Match matmul operations inside @matmul_tensors with test.attrC set.
pdl.pattern @pdl_target_attrC : benefit(1) {
  %args = operands
  %results = types
  %attr = attribute
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrC" = %attr}-> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@matmul_tensors](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @pdl_target_attrA
  tile %0 {sizes = [4, 4, 4]}
  %1 = match @pdl_target_attrC
  vectorize %1
}

// -----

// CHECK-LABEL: @vectorize_one
func @vectorize_one(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>,
  %arg3: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // CHECK: vector.contract
  %0 = linalg.matmul {test.attrA}
                     ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  // CHECK: linalg.matmul
  %1 = linalg.matmul ins(%arg0, %0: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg3: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

pdl.pattern @pdl_target : benefit(1) {
  %args = operands
  %results = types
  %attr = attribute
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrA" = %attr}-> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@vectorize_one](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @pdl_target
  vectorize %0
}


// -----

// CHECK-LABEL: @vectorize_all
func @vectorize_all(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>,
  %arg3: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // CHECK: vector.contract
  %0 = linalg.matmul {test.attrA}
                     ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  // CHECK: vector.contract
  %1 = linalg.matmul ins(%arg0, %0: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg3: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

iree_linalg_transform.sequence {
  vectorize
}
