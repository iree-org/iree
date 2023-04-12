// RUN: iree-opt %s --iree-stablehlo-to-linalg --split-input-file \
// RUN:   --canonicalize | FileCheck --enable-var-scope=false %s

func.func @dot_general(%arg0: tensor<?x?x?xf32>,
                  %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    someattr
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
// The iterations are (Batch Dim, LHS Other Dim, RHS Other dim, Contracting Dim)
// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0)>
// Output is the iterators excluding contracting
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: func @dot_general(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>)
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C0]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]])
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// Only contracting dims are reductions
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?xf32>)
// CHECK-SAME: {someattr}
// CHECK:   ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
// CHECK:     %[[MUL:.*]] = arith.mulf %[[ARG2]], %[[ARG3]] : f32
// CHECK:     %[[SUM:.*]] = arith.addf %[[ARG4]], %[[MUL]] : f32
// CHECK:     linalg.yield %[[SUM]] : f32
// CHECK: } -> tensor<?x?x?xf32>

// -----

func.func @dot_general_unsigned(%arg0: tensor<?x?x?xui32>,
                  %arg1: tensor<?x?x?xui32>) -> tensor<?x?x?xui32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    someattr
  } : (tensor<?x?x?xui32>, tensor<?x?x?xui32>) -> tensor<?x?x?xui32>
  func.return %0 : tensor<?x?x?xui32>
}

// CHECK-LABEL: func @dot_general_unsigned(
// CHECK: linalg.generic
// CHECK-SAME: ins({{.*}} : tensor<?x?x?xi32>, tensor<?x?x?xi32>)
// CHECK-SAME: outs({{.*}} : tensor<?x?x?xi32>)

// -----

func.func @dot_general_complex(%arg0: tensor<?x?x?xcomplex<f32>>,
                  %arg1: tensor<?x?x?xcomplex<f32>>) -> tensor<?x?x?xcomplex<f32>> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    someattr
  } : (tensor<?x?x?xcomplex<f32>>, tensor<?x?x?xcomplex<f32>>) -> tensor<?x?x?xcomplex<f32>>
  func.return %0 : tensor<?x?x?xcomplex<f32>>
}

// CHECK-LABEL: func @dot_general_complex(
// CHECK: linalg.generic
// CHECK: complex.mul
// CHECK: complex.add

// -----

func.func @dot_general_multiple_batch_dimensions(%arg0: tensor<3x4x2x4xi32>,
             %arg1: tensor<3x4x3x2xi32>) -> tensor<3x4x4x3xi32> {
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [3]>,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
    someattr
  } : (tensor<3x4x2x4xi32>, tensor<3x4x3x2xi32>) -> tensor<3x4x4x3xi32>
  return %0 : tensor<3x4x4x3xi32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @dot_general_multiple_batch_dimensions(
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<3x4x2x4xi32>, tensor<3x4x3x2xi32>)
// CHECK-SAME: outs({{.*}} : tensor<3x4x4x3xi32>)
// CHECK-SAME: {someattr}

// -----

func.func @dot_matmul(%arg0: tensor<2x3xf32>,
                 %arg1: tensor<3x?xf32>) -> tensor<2x?xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {someattr}
           : (tensor<2x3xf32>, tensor<3x?xf32>) -> tensor<2x?xf32>
  func.return %0 : tensor<2x?xf32>
}
// CHECK-LABEL: func @dot_matmul(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xf32>, %[[ARG1:.*]]: tensor<3x?xf32>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]])
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: {someattr}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xf32>, tensor<3x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xf32>)

// -----

func.func @dot_matmul_complex(%arg0: tensor<2x3xcomplex<f32>>,
                 %arg1: tensor<3x?xcomplex<f32>>) -> tensor<2x?xcomplex<f32>> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {someattr}
           : (tensor<2x3xcomplex<f32>>, tensor<3x?xcomplex<f32>>) -> tensor<2x?xcomplex<f32>>
  func.return %0 : tensor<2x?xcomplex<f32>>
}
// CHECK-LABEL: func @dot_matmul_complex(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xcomplex<f32>>, %[[ARG1:.*]]: tensor<3x?xcomplex<f32>>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: {someattr}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xcomplex<f32>>, tensor<3x?xcomplex<f32>>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xcomplex<f32>>)

// -----

func.func @dot_matmul_i8_i8_i32(%arg0: tensor<2x3xi8>,
                 %arg1: tensor<3x?xi8>) -> tensor<2x?xi32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x3xi8>,
                                   tensor<3x?xi8>) -> tensor<2x?xi32>
  func.return %0 : tensor<2x?xi32>
}
// CHECK-LABEL: func @dot_matmul_i8_i8_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xi8>, %[[ARG1:.*]]: tensor<3x?xi8>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xi8>, tensor<3x?xi8>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xi32>)

// -----

func.func @dot_matmul_i16_i16_i32(%arg0: tensor<2x3xi16>,
                 %arg1: tensor<3x?xi16>) -> tensor<2x?xi32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x3xi16>,
                                   tensor<3x?xi16>) -> tensor<2x?xi32>
  func.return %0 : tensor<2x?xi32>
}
// CHECK-LABEL: func @dot_matmul_i16_i16_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xi16>, %[[ARG1:.*]]: tensor<3x?xi16>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xi16>, tensor<3x?xi16>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xi32>)

// -----

func.func @dot_matmul_i32_i32_i32(%arg0: tensor<2x3xi32>,
                 %arg1: tensor<3x?xi32>) -> tensor<2x?xi32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<2x3xi32>,
                                   tensor<3x?xi32>) -> tensor<2x?xi32>
  func.return %0 : tensor<2x?xi32>
}
// CHECK-LABEL: func @dot_matmul_i32_i32_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xi32>, %[[ARG1:.*]]: tensor<3x?xi32>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xi32>, tensor<3x?xi32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xi32>)

// -----

func.func @dot_matvec(%arg0: tensor<?x3xf32>,
                 %arg1: tensor<3xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<?x3xf32>,
                                   tensor<3xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @dot_matvec(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x3xf32>, %[[ARG1:.*]]: tensor<3xf32>)
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]])
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matvec
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x3xf32>, tensor<3xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?xf32>)

// -----

func.func @dot_vecmat(%arg0: tensor<3xf32>,
                 %arg1: tensor<3x?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<3xf32>,
                                   tensor<3x?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @dot_vecmat(
// CHECK-SAME: %[[ARG0:.*]]: tensor<3xf32>, %[[ARG1:.*]]: tensor<3x?xf32>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]])
// CHECK: linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.vecmat
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<3xf32>, tensor<3x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?xf32>)

// -----

func.func @dot_dot(%arg0: tensor<?xf32>,
              %arg1: tensor<?xf32>) -> tensor<f32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: func @dot_dot(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK: %[[INIT:.*]] = tensor.empty()
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.dot
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?xf32>, tensor<?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<f32>)

// -----

func.func @dot_dot_unsigned(%arg0: tensor<?xui32>,
              %arg1: tensor<?xui32>) -> tensor<ui32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) : (tensor<?xui32>, tensor<?xui32>) -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-LABEL: func @dot_dot_unsigned(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?xui32>, %[[ARG1:.*]]: tensor<?xui32>)
// CHECK: %[[INIT:.*]] = tensor.empty()
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}outs(%[[INIT]]
// CHECK: linalg.dot
// CHECK-SAME: ins(%{{.*}} : tensor<?xi32>, tensor<?xi32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<i32>)
