// RUN: iree-opt -split-input-file -iree-vmla-conversion -cse %s | IreeFileCheck %s

// CHECK-LABEL: @single_reduction
func @single_reduction(%arg0: tensor<4x8xf32>) -> tensor<4xf32> attributes { sym_visibility = "private" } {
  // CHECK-DAG: %[[INIT:.+]] = vmla.constant dense<0.000000e+00> : tensor<f32> -> !vmla.buffer
  %cst = constant dense<0.000000e+00> : tensor<f32>
  //  CHECK-DAG: %[[SRC_SHAPE:.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[4,8]>
  //  CHECK-DAG: %[[INIT_SHAPE:.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[]>
  //  CHECK-DAG: %[[DST:.+]] = vmla.buffer.alloc
  //  CHECK-DAG: %[[DST_SHAPE:.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[4]>
  // CHECK-NEXT: vmla.reduce.sum
  // CEHCK-SAME: %arg0(%[[SRC_SHAPE]] : !shapex.ranked_shape<[4,8]>),
  // CHECK-SAME: %[[INIT]](%[[INIT_SHAPE]] : !shapex.ranked_shape<[]>),
  // CHECK-SAME: out %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[4]>)
  // CHECK-SaME: {dimension = 1 : i32} : f32
  %0 = "mhlo.reduce"(%arg0, %cst) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %1 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: return %[[DST]] : !vmla.buffer
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @multi_reduction
func @multi_reduction(%arg0 : tensor<4x8xf32>, %arg1 : tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>) attributes { sym_visibility = "private" } {
  //  CHECK-DAG: %[[CST0:.+]] = vmla.constant dense<0.000000e+00> : tensor<f32> -> !vmla.buffer
  %0 = constant dense<0.000000e+00> : tensor<f32>
  //  CHECK-DAG: %[[CST1:.+]] = vmla.constant dense<1.000000e+00> : tensor<f32> -> !vmla.buffer
  %1 = constant dense<1.000000e+00> : tensor<f32>
  // CHECK-DAG: %[[INPUT_SHAPE:.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[4,8]>
  // CHECK-DAG: %[[SCALAR_SHAPE:.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[]>
  // CHECK-DAG: %[[RESULT_SHAPE:.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[4]>
  // CHECK-DAG: %[[RET_SIZE:.+]] = muli
  // CHECK-DAG: %[[RET0:.+]] = vmla.buffer.alloc byte_length = %[[RET_SIZE]] : !vmla.buffer
  // CHECK-NEXT: vmla.reduce.sum
  // CEHCK-SAME: %arg0(%[[INPUT_SHAPE]] : !shapex.ranked_shape<[4,8]>),
  // CHECK-SAME: %[[CST0]](%[[SCALAR_SHAPE]] : !shapex.ranked_shape<[]>),
  // CHECK-SAME: out %[[RET0]](%[[RESULT_SHAPE]] : !shapex.ranked_shape<[4]>)
  // CHECK-SaME: {dimension = 1 : i32} : f32
  // CHECK-NEXT: %[[RET1:.+]] = vmla.buffer.alloc byte_length = %[[RET_SIZE]] : !vmla.buffer
  // CHECK-NEXT: vmla.reduce.sum
  // CEHCK-SAME: %arg1(%[[INPUT_SHAPE]] : !shapex.ranked_shape<[4,8]>),
  // CHECK-SAME: %[[CST1]](%[[SCALAR_SHAPE]] : !shapex.ranked_shape<[]>),
  // CHECK-SAME: out %[[RET1]](%[[RESULT_SHAPE]] : !shapex.ranked_shape<[4]>)
  // CHECK-SaME: {dimension = 1 : i32} : f32
  %2, %3 = "mhlo.reduce"(%arg0, %arg1, %0, %1) ( {
  ^bb0(%arg0_lhs : tensor<f32>, %arg1_lhs : tensor<f32>, %arg0_rhs : tensor<f32>, %arg1_rhs : tensor<f32>):
    %4 = mhlo.add %arg0_lhs, %arg0_rhs : tensor<f32>
    %5 = mhlo.add %arg1_lhs, %arg1_rhs : tensor<f32>
    "mhlo.return"(%4, %5) : (tensor<f32>, tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
  // CHECK-NEXT: return %[[RET0]], %[[RET1]] : !vmla.buffer, !vmla.buffer
  return %2, %3 : tensor<4xf32>, tensor<4xf32>
}
