// RUN: iree-opt -split-input-file -iree-vmla-conversion -cse %s | IreeFileCheck %s

// CHECK-LABEL: @single_reduction
func @single_reduction(%arg0: tensor<4x8xf32>) -> tensor<4xf32> attributes { sym_visibility = "private" } {
  // CHECK-DAG: [[INIT:%.+]] = "vmla.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> !vmla.buffer
  %cst = constant dense<0.000000e+00> : tensor<f32>
  //  CHECK-DAG: [[SRC_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[4,8]>
  //  CHECK-DAG: [[INIT_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[]>
  //  CHECK-DAG: [[DST:%.+]] = "vmla.buffer.alloc"
  //  CHECK-DAG: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[4]>
  // CHECK-NEXT: "vmla.reduce.sum"(%arg0, [[SRC_SHAPE]], [[INIT]], [[INIT_SHAPE]], [[DST]], [[DST_SHAPE]]) {dimension = 1 : i32, element_type = f32} : (!vmla.buffer, !shapex.ranked_shape<[4,8]>, !vmla.buffer, !shapex.ranked_shape<[]>, !vmla.buffer, !shapex.ranked_shape<[4]>) -> ()
  %0 = "xla_hlo.reduce"(%arg0, %cst) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):	// no predecessors
    %1 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    "xla_hlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: return [[DST]] : !vmla.buffer
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @multi_reduction
func @multi_reduction(%arg0 : tensor<4x8xf32>, %arg1 : tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>) attributes { sym_visibility = "private" } {
  //  CHECK-DAG: [[CST0:%.+]] = "vmla.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> !vmla.buffer
  %0 = constant dense<0.000000e+00> : tensor<f32>
  //  CHECK-DAG: [[CST1:%.+]] = "vmla.constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> !vmla.buffer
  %1 = constant dense<1.000000e+00> : tensor<f32>
  //  CHECK-DAG: [[INPUT_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[4,8]>
  //  CHECK-DAG: [[SCALAR_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[]>
  //  CHECK-DAG: [[RESULT_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[4]>
  //  CHECK-DAG: [[RET_SIZE:%.+]] = muli
  // CHECK-NEXT: [[RET0:%.+]] = "vmla.buffer.alloc"([[RET_SIZE]])
  // CHECK-NEXT: "vmla.reduce.sum"(%arg0, [[INPUT_SHAPE]], [[CST0]], [[SCALAR_SHAPE]], [[RET0]], [[RESULT_SHAPE]]) {dimension = 1 : i32, element_type = f32} : (!vmla.buffer, !shapex.ranked_shape<[4,8]>, !vmla.buffer, !shapex.ranked_shape<[]>, !vmla.buffer, !shapex.ranked_shape<[4]>) -> ()
  // CHECK-NEXT: [[RET1:%.+]] = "vmla.buffer.alloc"([[RET_SIZE]])
  // CHECK-NEXT: "vmla.reduce.sum"(%arg1, [[INPUT_SHAPE]], [[CST1]], [[SCALAR_SHAPE]], [[RET1]], [[RESULT_SHAPE]]) {dimension = 1 : i32, element_type = f32} : (!vmla.buffer, !shapex.ranked_shape<[4,8]>, !vmla.buffer, !shapex.ranked_shape<[]>, !vmla.buffer, !shapex.ranked_shape<[4]>) -> ()
  %2, %3 = "xla_hlo.reduce"(%arg0, %arg1, %0, %1) ( {
  ^bb0(%arg0_lhs : tensor<f32>, %arg1_lhs : tensor<f32>, %arg0_rhs : tensor<f32>, %arg1_rhs : tensor<f32>):
    %4 = xla_hlo.add %arg0_lhs, %arg0_rhs : tensor<f32>
    %5 = xla_hlo.add %arg1_lhs, %arg1_rhs : tensor<f32>
    "xla_hlo.return"(%4, %5) : (tensor<f32>, tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
  // CHECK-NEXT: return [[RET0]], [[RET1]] : !vmla.buffer, !vmla.buffer
  return %2, %3 : tensor<4xf32>, tensor<4xf32>
}
