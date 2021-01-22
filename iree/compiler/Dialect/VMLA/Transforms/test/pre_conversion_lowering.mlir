// RUN: iree-opt -split-input-file -iree-vmla-pre-conversion-lowering %s | IreeFileCheck %s

// -----

// CHECK-LABEL: func @dot_general_float
func @dot_general_float(%arg0: tensor<3x4xf32>, %arg1: tensor<4x5xf32>) -> tensor<3x5xf32> {
  // CHECK: vmla.batch.matmul
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = {
    lhs_batching_dimensions = dense<[]> : tensor<0xi64>,
    lhs_contracting_dimensions = dense<[1]> : tensor<1xi64>,
    rhs_batching_dimensions = dense<[]> : tensor<0xi64>,
    rhs_contracting_dimensions = dense<[0]> : tensor<1xi64>
  }} : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// -----

// CHECK-LABEL: func @dot_mixed_element_types
func @dot_mixed_element_types(%arg0: tensor<2x3xi8>, %arg1: tensor<3x2xi16>) -> tensor<2x2xi32> {
  // CHECK: vmla.batch.matmul.pseudo %{{[a-zA-Z0-9$._-]+}}, %{{[a-zA-Z0-9$._-]+}} : (tensor<?x?x?xi8>, tensor<?x?x?xi16>) -> tensor<?x?x?xi32>
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x3xi8>, tensor<3x2xi16>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func private @sort
func private @sort(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-DAG: [[SORT:%.+]] = vmla.sort.pseudo %arg0
  // CHECK-DAG: [[GATHER:%.+]] = "mhlo.torch_index_select"(%arg0, [[SORT]]) {batch_dims = 0 : i64, dim = 0 : i64}
  %sort = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %compare = "mhlo.compare"(%arg1, %arg2) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<4xf32>) -> tensor<4xf32>

  // CHECK: return [[GATHER]]
  return %sort : tensor<4xf32>
}

// -----

// CHECK-LABEL: func private @sort
func private @sort_2d(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-DAG: [[SORT:%.+]] = vmla.sort.pseudo %arg0
  // CHECK-DAG: [[GATHER:%.+]] = "mhlo.torch_index_select"(%arg0, [[SORT]]) {batch_dims = 1 : i64, dim = 1 : i64}
  %sort = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %compare = "mhlo.compare"(%arg1, %arg2) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = false} : (tensor<4x4xf32>) -> tensor<4x4xf32>

  // CHECK return [[GATHER]]
  return %sort : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @broadcast_in_dim
func @broadcast_in_dim(%arg0: tensor<3xf32>) -> tensor<4x3xf32> {
  // CHECK: "shapex.ranked_broadcast_in_dim"(%arg0, %rs4_3)
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<4x3xf32>
  return %0 : tensor<4x3xf32>
}

// -----

// CHECK-LABEL: func @ranked_broadcast_in_dim
func @ranked_broadcast_in_dim(%arg0: tensor<3xf32>) -> tensor<5x6x3xf32> {
  // CHECK: "shapex.ranked_broadcast_in_dim"(%arg0, %rs5_6_3)
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[5, 6]> : tensor<2xi64>} : (tensor<3xf32>) -> tensor<5x6x3xf32>
  return %0 : tensor<5x6x3xf32>
}

// -----

// CHECK-LABEL: func private @fft
func private @fft(%arg0: tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>> {
  // CHECK-DAG: [[REAL:%.+]] = "mhlo.real"(%arg0)
  // CHECK-DAG: [[IMAG:%.+]] = "mhlo.imag"(%arg0)
  // CHECK-DAG: [[REAL_OUT:%.+]], [[IMAG_OUT:%.+]] = vmla.fft.pseudo [[REAL]], [[IMAG]]
  // CHECK: "mhlo.complex"([[REAL_OUT]], [[IMAG_OUT]])
  %0 = "mhlo.fft"(%arg0) {fft_length = dense<8> : tensor<1xi64>, fft_type = "FFT"} : (tensor<8xcomplex<f32>>) -> tensor<8xcomplex<f32>>
  return %0 : tensor<8xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_multiply
func @complex_multiply(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  // CHECK-NOT: "mhlo.complex"
  %0 = "mhlo.complex"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xcomplex<f32>>

  // CHECK-DAG: [[V1:%.+]] = mhlo.multiply %arg0, %arg0
  // CHECK-DAG: [[V2:%.+]] = mhlo.multiply %arg1, %arg1
  // CHECK-DAG: [[V3:%.+]] = mhlo.subtract [[V1]], [[V2]]
  %1 = "mhlo.multiply"(%0, %0) : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>>
  %2 = "mhlo.real"(%1) : (tensor<3xcomplex<f32>>) -> tensor<3xf32>

  // CHECK: return [[V3]]
  return %2 : tensor<3xf32>
}
