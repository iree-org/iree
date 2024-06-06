// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-tensor-ops-to-linalg))" --split-input-file %s | FileCheck %s

// first all tests where i assume no reduced dims
func.func @basic(%arg0: tensor<2x1xi64>, %arg1: tensor<16x4xf32>) -> tensor<2x1x4xf32> {
  %gather = tensor.gather %arg1[%arg0] gather_dims([0]) : (tensor<16x4xf32>, tensor<2x1xi64>) -> tensor<2x1x4xf32>
  return %gather : tensor<2x1x4xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func.func @basic
// CHECK-NOT: tensor.gather
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK: arith.index_cast
// CHECK: linalg.index 2
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @dyn_shape(%arg0: tensor<?x1xi64>, %arg1: tensor<16x4xf32>) -> tensor<?x1x4xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0]) : (tensor<16x4xf32>, tensor<?x1xi64>) -> tensor<?x1x4xf32>
return %gather : tensor<?x1x4xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func.func @dyn_shape
// CHECK-NOT: tensor.gather
// CHECK: arith.constant
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK: arith.index_cast
// CHECK: linalg.index 2
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @many_dims(%arg0: tensor<5x10x2x2x1xi64>, %arg1: tensor<16x4xf32>) -> tensor<5x10x2x2x1x4xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0]) : (tensor<16x4xf32>, tensor<5x10x2x2x1xi64>) -> tensor<5x10x2x2x1x4xf32>
return %gather : tensor<5x10x2x2x1x4xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK: func.func @many_dims
// CHECK-NOT: tensor.gather
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK: arith.index_cast
// CHECK: linalg.index 5
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @gather_1x1x1x1(%arg0: tensor<1x3xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x1x1x1xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1, 2]) : (tensor<3x3x3xf32>, tensor<1x3xi64>) -> tensor<1x1x1x1xf32>
return %gather : tensor<1x1x1x1xf32>
}
// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, 1)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, 2)>
// CHECK: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @gather_1x1x1x1
// CHECK-NOT: tensor.gather
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @gather_1x1x1x3(%arg0: tensor<1x2xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x1x1x3xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1]) : (tensor<3x3x3xf32>, tensor<1x2xi64>) -> tensor<1x1x1x3xf32>
return %gather : tensor<1x1x1x3xf32>
}
// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, 1)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func @gather_1x1x1x3
// CHECK-NOT: tensor.gather
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: linalg.index 3
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

// tests with reduced dims
func.func @gather_1(%arg0: tensor<1x3xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1, 2]) : (tensor<3x3x3xf32>, tensor<1x3xi64>) -> tensor<1xf32>
return %gather : tensor<1xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0) -> (d0, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0) -> (d0, 1)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0) -> (d0, 2)>
// CHECK: #[[MAP3:.*]] = affine_map<(d0) -> (d0)>
// CHECK: func @gather_1
// CHECK-NOT: tensor.gather
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @gather_1x3(%arg0: tensor<1x2xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1]) : (tensor<3x3x3xf32>, tensor<1x2xi64>) -> tensor<1x3xf32>
return %gather : tensor<1x3xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, 1)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @gather_1x3
// CHECK-NOT: tensor.gather
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: linalg.index 1
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----


func.func @gather_1x3x3_dim_1(%arg0: tensor<1x1xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3x3xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([1]) : (tensor<3x3x3xf32>, tensor<1x1xi64>) -> tensor<1x3x3xf32>
return %gather : tensor<1x3x3xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @gather_1x3x3_dim_1
// CHECK-NOT: tensor.gather
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK: %[[DIM1:.+]] = arith.index_cast
// CHECK-DAG: %[[DIM0:.+]] = linalg.index 1 : index
// CHECK-DAG: %[[DIM2:.+]] = linalg.index 2 : index
// CHECK: tensor.extract %arg1[%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @gather_1x3x3_dim_2(%arg0: tensor<1x1xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3x3xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([2]) : (tensor<3x3x3xf32>, tensor<1x1xi64>) -> tensor<1x3x3xf32>
return %gather : tensor<1x3x3xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @gather_1x3x3_dim_2
// CHECK-NOT: tensor.gather
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK: %[[DIM2:.+]] = arith.index_cast
// CHECK-DAG: %[[DIM0:.+]] = linalg.index 1 : index
// CHECK-DAG: %[[DIM1:.+]] = linalg.index 2 : index
// CHECK: tensor.extract %arg1[%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @gather_1x3_dim_1_2(%arg0: tensor<1x2xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([1, 2]) : (tensor<3x3x3xf32>, tensor<1x2xi64>) -> tensor<1x3xf32>
return %gather : tensor<1x3xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, 1)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @gather_1x3_dim_1_2
// CHECK-NOT: tensor.gather
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK: %[[DIM1:.+]] = arith.index_cast
// CHECK: %[[DIM2:.+]] = arith.index_cast
// CHECK-DAG: %[[DIM0:.+]] = linalg.index 1 : index
// CHECK: tensor.extract %arg1[%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @source_1(%arg0: tensor<1x2xi64>, %arg1: tensor<1x1x1xf32>) -> tensor<1x1xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([1, 2]) : (tensor<1x1x1xf32>, tensor<1x2xi64>) -> tensor<1x1xf32>
return %gather : tensor<1x1xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, 1)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func @source_1
// CHECK-NOT: tensor.gather
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK: %[[DIM1:.+]] = arith.index_cast
// CHECK: %[[DIM2:.+]] = arith.index_cast
// CHECK-DAG: %[[DIM0:.+]] = linalg.index 1 : index
// CHECK: tensor.extract %arg1[%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----
