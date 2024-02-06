// RUN: iree-opt --split-input-file --iree-indexing-test-index-range-analysis --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func @constant_add
func.func @constant_add() -> index {
  %cst0 = arith.constant 25 : index
  %cst1 = arith.constant 7 : index
  %add = arith.addi %cst0, %cst1 : index

  // CHECK: analysis = "index-range: <[32, 32], 32>"
  %test = "iree_unregistered.test_intrange"(%add) : (index) -> index
  return %test : index
}

// -----

// CHECK-LABEL: func @range_add
func.func @range_add(%arg0: index, %arg1: index) -> index {
  %cst = arith.constant 15 : index
  %asserted = indexing.assert.aligned_range %arg0 range(-30, 480) align(30) : index
  %add = arith.addi %cst, %asserted : index

  %asserted2 = indexing.assert.aligned_range %arg1 range(-1080, 5) align(5) : index
  %add2 = arith.addi %add, %asserted2 : index

  // CHECK: analysis = "index-range: <[-1095, 500], 5>"
  %test = "iree_unregistered.test_intrange"(%add2) : (index) -> index
  return %test : index
}

// -----

// CHECK-LABEL: func @range_unbounded_add
func.func @range_unbounded_add(%arg0: index, %arg1: index) {
  %asserted = indexing.assert.aligned_range %arg0 range(UNBOUNDED, 20) : index
  %asserted2 = indexing.assert.aligned_range %arg1 range(-10, UNBOUNDED) : index
  %add = arith.addi %asserted, %asserted2 : index
  %add1 = arith.addi %asserted, %asserted : index
  %add2 = arith.addi %asserted2, %asserted2 : index

  // CHECK: analysis = "index-range: <[UNBOUNDED, UNBOUNDED], 1>"
  %test = "iree_unregistered.test_intrange"(%add) : (index) -> index
  // CHECK: analysis = "index-range: <[UNBOUNDED, 40], 1>"
  %test1 = "iree_unregistered.test_intrange"(%add1) : (index) -> index
  // CHECK: analysis = "index-range: <[-20, UNBOUNDED], 1>"
  %test2 = "iree_unregistered.test_intrange"(%add2) : (index) -> index
  return
}

// -----

// CHECK-LABEL: func @range_difference
func.func @range_difference(%arg0: index, %arg1: index) -> index {
  %asserted = indexing.assert.aligned_range %arg0 range(-20, 20) align(19) : index
  %asserted2 = indexing.assert.aligned_range %arg1 range(-80, 80) align(38) : index
  %sub = arith.subi %asserted, %asserted2 : index

  // CHECK: analysis = "index-range: <[-95, 95], 19>"
  %test = "iree_unregistered.test_intrange"(%sub) : (index) -> index
  return %test : index
}

// -----

// CHECK-LABEL: func @range_unbounded_difference
func.func @range_unbounded_difference(%arg0: index, %arg1: index) {
  %asserted = indexing.assert.aligned_range %arg0 range(UNBOUNDED, 20) : index
  %asserted2 = indexing.assert.aligned_range %arg1 range(-10, UNBOUNDED) : index
  %sub = arith.subi %asserted, %asserted2 : index
  %sub1 = arith.subi %asserted2, %asserted : index
  %sub2 = arith.subi %asserted, %asserted : index

  // CHECK: analysis = "index-range: <[UNBOUNDED, 30], 1>"
  %test = "iree_unregistered.test_intrange"(%sub) : (index) -> index
  // CHECK: analysis = "index-range: <[-30, UNBOUNDED], 1>"
  %test1 = "iree_unregistered.test_intrange"(%sub1) : (index) -> index
  // CHECK: analysis = "index-range: <[UNBOUNDED, UNBOUNDED], 1>"
  %test2 = "iree_unregistered.test_intrange"(%sub2) : (index) -> index
  return
}

// -----

// CHECK-LABEL: func @range_product
func.func @range_product(%arg0: index, %arg1: index, %arg2: index)
    -> (index, index, index, index, index) {
  %asserted0 = indexing.assert.aligned_range %arg0 range(18, 26) align(2) : index
  %asserted1 = indexing.assert.aligned_range %arg1 range(-9, 21) align(3) : index
  %asserted2 = indexing.assert.aligned_range %arg2 range(-60, -15) align(5) : index
  %mul0 = arith.muli %asserted0, %asserted0 : index
  %mul1 = arith.muli %asserted0, %asserted1 : index
  %mul2 = arith.muli %asserted0, %asserted2 : index
  %mul3 = arith.muli %asserted1, %asserted2 : index
  %mul4 = arith.muli %asserted2, %asserted2 : index

  // CHECK: analysis = "index-range: <[324, 676], 4>"
  %test0 = "iree_unregistered.test_intrange"(%mul0) : (index) -> index
  // CHECK: analysis = "index-range: <[-234, 546], 6>"
  %test1 = "iree_unregistered.test_intrange"(%mul1) : (index) -> index
  // CHECK: analysis = "index-range: <[-1560, -270], 10>"
  %test2 = "iree_unregistered.test_intrange"(%mul2) : (index) -> index
  // CHECK: analysis = "index-range: <[-1260, 540], 15>"
  %test3 = "iree_unregistered.test_intrange"(%mul3) : (index) -> index
  // CHECK: analysis = "index-range: <[225, 3600], 25>"
  %test4 = "iree_unregistered.test_intrange"(%mul4) : (index) -> index
  return %test0, %test1, %test2, %test3, %test4 : index, index, index, index, index
}

// -----

// CHECK-LABEL: func @range_unbounded_product
func.func @range_unbounded_product(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
  %asserted0 = indexing.assert.aligned_range %arg0 range(UNBOUNDED, 1) : index
  %asserted1 = indexing.assert.aligned_range %arg1 range(UNBOUNDED, -2) : index
  %asserted2 = indexing.assert.aligned_range %arg2 range(3, UNBOUNDED) : index
  %asserted3 = indexing.assert.aligned_range %arg3 range(-5, -2) : index
  %mul0 = arith.muli %asserted0, %asserted0 : index
  %mul1 = arith.muli %asserted1, %asserted0 : index
  %mul2 = arith.muli %asserted1, %asserted1 : index
  %mul3 = arith.muli %asserted2, %asserted1 : index
  %mul4 = arith.muli %asserted2, %asserted2 : index
  %mul5 = arith.muli %asserted3, %asserted1 : index
  %mul6 = arith.muli %asserted3, %asserted2 : index

  // CHECK: analysis = "index-range: <[UNBOUNDED, UNBOUNDED], 1>"
  %test0 = "iree_unregistered.test_intrange"(%mul0) : (index) -> index
  // CHECK: analysis = "index-range: <[UNBOUNDED, UNBOUNDED], 1>"
  %test1 = "iree_unregistered.test_intrange"(%mul1) : (index) -> index
  // CHECK: analysis = "index-range: <[4, UNBOUNDED], 1>"
  %test2 = "iree_unregistered.test_intrange"(%mul2) : (index) -> index
  // CHECK: analysis = "index-range: <[UNBOUNDED, -6], 1>"
  %test3 = "iree_unregistered.test_intrange"(%mul3) : (index) -> index
  // CHECK: analysis = "index-range: <[9, UNBOUNDED], 1>"
  %test4 = "iree_unregistered.test_intrange"(%mul4) : (index) -> index
  // CHECK: analysis = "index-range: <[4, UNBOUNDED], 1>"
  %test5 = "iree_unregistered.test_intrange"(%mul5) : (index) -> index
  // CHECK: analysis = "index-range: <[UNBOUNDED, -6], 1>"
  %test6 = "iree_unregistered.test_intrange"(%mul6) : (index) -> index
  return
}

// -----

// CHECK-LABEL: func @shaped_dims
func.func @shaped_dims(%arg0: tensor<2x?x?xf32>) -> tensor<2x?x?xf32> {
  %asserted = indexing.assert.dim_range %arg0[1] range(4, 40) align(4) : tensor<2x?x?xf32>
  %asserted2 = indexing.assert.dim_range %asserted[2] range(3, 9) align(3) : tensor<2x?x?xf32>

  // CHECK: analysis = "index-range: <[4, 40], 4>, <[3, 9], 3>"
  %test = "iree_unregistered.test_dimranges"(%asserted2) : (tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  return %test : tensor<2x?x?xf32>
}

// -----

// CHECK-LABEL: func @destination_passing
func.func @destination_passing(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %asserted = indexing.assert.dim_range %arg0[0] range(2, 3) align(1) : tensor<?xf32>

  %a = linalg.add ins(%arg0, %arg0: tensor<?xf32>, tensor<?xf32>) outs(%asserted: tensor<?xf32>) -> tensor<?xf32>

  %inserted = tensor.insert %cst into %a[%c0] : tensor<?xf32>

  %s = tensor.dim %arg0, %c0 : tensor<?xf32>
  %inserted_slice = tensor.insert_slice %arg0 into %inserted[0] [%s] [1] : tensor<?xf32> into tensor<?xf32>

  // CHECK: analysis = "index-range: <[2, 3], 1>
  %test = "iree_unregistered.test_dimranges"(%inserted_slice) : (tensor<?xf32>) -> tensor<?xf32>
  return %test : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @dim_upper_bound
func.func @dim_upper_bound(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %asserted = indexing.assert.dim_range %arg0[0] range(UNBOUNDED, 3) align(1) : tensor<?xf32>
  // Dims can never be negative.
  // CHECK: analysis = "index-range: <[0, 3], 1>
  %test = "iree_unregistered.test_dimranges"(%asserted) : (tensor<?xf32>) -> tensor<?xf32>
  return %test : tensor<?xf32>
}
