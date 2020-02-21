// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @broadcast_in_dim_2D_3D
func @broadcast_in_dim_2D_3D() -> tensor<3x2x4xi32> {
  // CHECK-DAG: [[SRC:%.+]] = "vmla.constant"
  // CHECK-DAG: [[SRC_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[2,4],i32>
  // CHECK-DAG: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[3,2,4],i32>
  // CHECK-DAG: [[DST_SIZE:%.+]] = constant 96 : i32
  %input = constant dense<[[1, 2, 3, 4], [5, 6, 7, 8]]> : tensor<2x4xi32>
  // CHECK-NEXT: [[DST:%.+]] = "vmla.buffer.alloc"([[DST_SIZE]])
  // CHECK-NEXT: "vmla.tile"([[SRC]], [[SRC_SHAPE]], [[DST]], [[DST_SHAPE]]) {element_type = i32}
  %0 = "xla_hlo.broadcast_in_dim"(%input) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  // CHECK-NEXT: return [[DST]] : !vmla.buffer
  return %0 : tensor<3x2x4xi32>
}

// -----

// CHECK-LABEL: @broadcast_in_dim_3D_scalar
func @broadcast_in_dim_3D_scalar() -> tensor<3x2x4xi32> {
  // CHECK-DAG: [[SRC:%.+]] = "vmla.constant"
  // CHECK-DAG: [[SRC_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[],i32>
  // CHECK-DAG: [[DST_SHAPE:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[3,2,4],i32>
  // CHECK-DAG: [[DST_SIZE:%.+]] = constant 96 : i32
  %input = constant dense<42> : tensor<i32>
  // CHECK-NEXT: [[DST:%.+]] = "vmla.buffer.alloc"([[DST_SIZE]])
  // CHECK-NEXT: "vmla.broadcast"([[SRC]], [[SRC_SHAPE]], [[DST]], [[DST_SHAPE]]) {element_type = i32}
  %0 = "xla_hlo.broadcast_in_dim"(%input) : (tensor<i32>) -> tensor<3x2x4xi32>
  // CHECK-NEXT: return [[DST]] : !vmla.buffer
  return %0 : tensor<3x2x4xi32>
}
