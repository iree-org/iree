// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @concatenate_0
func @concatenate_0(%arg0 : tensor<2x2xi32>) -> (tensor<2x5xi32>) attributes { sym_visibility = "private" } {
  // CHECK-SAME: %[[ARG0:.+]]:
  // CHECK-DAG: %[[ARG1:.+]] = vmla.constant {{.+}} tensor<2x3xi32>
  %c0 = constant dense<[[5, 6, 7], [8, 9, 10]]> : tensor<2x3xi32>
  // CHECK: %[[DST:.+]] = vmla.buffer.alloc byte_length = %c40 : !vmla.buffer
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[ARG0]](%rs2_2 : !shapex.ranked_shape<[2,2]>),
  // CHECK-SAME: src_indices = [%c0, %c0],
  // CHECK-SAME: out %[[DST]](%rs2_5 : !shapex.ranked_shape<[2,5]>),
  // CHECK-SAME: dst_indices = [%c0, %c0], lengths = [%c2, %c2] : i32
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[ARG1]](%rs2_3 : !shapex.ranked_shape<[2,3]>),
  // CHECK-SAME: src_indices = [%c0, %c0],
  // CHECK-SAME: out %[[DST]](%rs2_5 : !shapex.ranked_shape<[2,5]>),
  // CHECK-SAME: dst_indices = [%c0, %c2], lengths = [%c2, %c3] : i32
  %0 = "mhlo.concatenate"(%arg0, %c0) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x5xi32>
  // CHECK-NEXT: return %[[DST]]
  return %0: tensor<2x5xi32>
}

// -----

// CHECK-LABEL: @concatenate_1
func @concatenate_1(%arg0: tensor<2x3xi32>) -> (tensor<2x5xi32>) attributes { sym_visibility = "private" } {
  // CHECK-SAME: %[[ARG0:.+]]:
  // CHECK-DAG: %[[ARG1:.+]] = vmla.constant {{.+}} tensor<2x2xi32>
  %c0 = constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  // CHECK: %[[DST:.+]] = vmla.buffer.alloc byte_length = %c40 : !vmla.buffer
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[ARG0]](%rs2_3 : !shapex.ranked_shape<[2,3]>),
  // CHECK-SAME: src_indices = [%c0, %c0],
  // CHECK-SAME: out %[[DST]](%rs2_5 : !shapex.ranked_shape<[2,5]>),
  // CHECK-SAME: dst_indices = [%c0, %c0], lengths = [%c2, %c3] : i32
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[ARG1]](%rs2_2 : !shapex.ranked_shape<[2,2]>),
  // CHECK-SAME: src_indices = [%c0, %c0],
  // CHECK-SAME: out %[[DST]](%rs2_5 : !shapex.ranked_shape<[2,5]>),
  // CHECK-SAME: dst_indices = [%c0, %c3], lengths = [%c2, %c2] : i32
  %0 = "mhlo.concatenate"(%arg0, %c0) {dimension = 1} : (tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x5xi32>
  // CHECK-NEXT: return %[[DST]]
  return %0: tensor<2x5xi32>
}

// -----

// CHECK-LABEL: @concatenate_2
func @concatenate_2(%arg0: tensor<2x2xi32>) -> (tensor<2x7xi32>) attributes { sym_visibility = "private" } {
  // CHECK-SAME: %[[ARG0:.+]]:
  // CHECK-DAG: %[[ARG1:.+]] = vmla.constant {{.+}} tensor<2x3xi32>
  %c0 = constant dense<[[5, 6, 7], [8, 9, 10]]> : tensor<2x3xi32>
  // CHECK-DAG: %[[ARG2:.+]] = vmla.constant {{.+}} tensor<2x2xi32>
  %c1 = constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>
  // CHECK: %[[DST:.+]] = vmla.buffer.alloc byte_length = %c56 : !vmla.buffer
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[ARG0]](%rs2_2 : !shapex.ranked_shape<[2,2]>),
  // CHECK-SAME: src_indices = [%c0, %c0],
  // CHECK-SAME: out %[[DST]](%rs2_7 : !shapex.ranked_shape<[2,7]>),
  // CHECK-SAME: dst_indices = [%c0, %c0], lengths = [%c2, %c2] : i32
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[ARG1]](%rs2_3 : !shapex.ranked_shape<[2,3]>),
  // CEHCK-SAME: src_indices = [%c0, %c0],
  // CHECK-SAME: out %[[DST]](%rs2_7 : !shapex.ranked_shape<[2,7]>),
  // CHECK-SAME: dst_indices = [%c0, %c2], lengths = [%c2, %c3] : i32
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[ARG2]](%rs2_2 : !shapex.ranked_shape<[2,2]>),
  // CHECK-SAME: src_indices = [%c0, %c0],
  // CHECK-SAME: out %[[DST]](%rs2_7 : !shapex.ranked_shape<[2,7]>),
  // CHECK-SAME: dst_indices = [%c0, %c5], lengths = [%c2, %c2] : i32
  %0 = "mhlo.concatenate"(%arg0, %c0, %c1) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x7xi32>
  // CHECK-NEXT: return %[[DST]]
  return %0: tensor<2x7xi32>
}

// -----

// CHECK-LABEL: @concatenate_3
func @concatenate_3(%arg0: tensor<2x2xi32>) -> (tensor<4x2xi32>) attributes { sym_visibility = "private" } {
  // CHECK-SAME: %[[ARG0:.+]]:
  // CHECK-DAG: %[[ARG1:.+]] = vmla.constant {{.+}} tensor<2x2xi32>
  %c0 = constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>
  // CHECK: %[[DST:.+]] = vmla.buffer.alloc byte_length = %c32 : !vmla.buffer
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[ARG0]](%rs2_2 : !shapex.ranked_shape<[2,2]>),
  // CHECK-SAME: src_indices = [%c0, %c0],
  // CHECK-SAME: out %[[DST]](%rs4_2 : !shapex.ranked_shape<[4,2]>),
  // CHECK-SAME: dst_indices = [%c0, %c0], lengths = [%c2, %c2] : i32
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[ARG1]](%rs2_2 : !shapex.ranked_shape<[2,2]>),
  // CHECK-SAME: src_indices = [%c0, %c0],
  // CHECK-SAME: out %[[DST]](%rs4_2 : !shapex.ranked_shape<[4,2]>),
  // CHECK-SAME: dst_indices = [%c2, %c0], lengths = [%c2, %c2] : i32
  %0 = "mhlo.concatenate"(%arg0, %c0) {dimension = 0} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<4x2xi32>
  // CHECK-NEXT: return %[[DST]]
  return %0: tensor<4x2xi32>
}
