// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-tensor-ops-to-linalg))" --split-input-file %s | FileCheck %s





// CHECK-DAG:  affine_map<(d0, d1, d2) -> (d0, 0)>
// CHECK-DAG:  affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, 0)>
// CHECK-DAG:  affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-DAG:  affine_map<(d0, d1, d2, d3) -> (d0, 0)>
// CHECK-DAG:  affine_map<(d0, d1, d2, d3) -> (d0, 1)>
// CHECK-DAG:  affine_map<(d0, d1, d2, d3) -> (d0, 2)>
// CHECK-DAG:  affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG:  affine_map<(d0) -> (d0, 0)>
// CHECK-DAG:  affine_map<(d0) -> (d0, 1)>
// CHECK-DAG:  affine_map<(d0) -> (d0, 2)>
// CHECK-DAG:  affine_map<(d0) -> (d0)>
// CHECK-DAG:  affine_map<(d0, d1) -> (d0, 0)>
// CHECK-DAG:  affine_map<(d0, d1) -> (d0, 1)>
// CHECK-DAG:  affine_map<(d0, d1) -> (d0, d1)>

// first all tests where i assume no reduced dims

// CHECK-NOT: tensor.gather


// CHECK-LABEL: func.func @basic
func.func @basic(%arg0: tensor<2x1xi64>, %arg1: tensor<16x4xf32>) -> tensor<2x1x4xf32> {
// CHECK: %[[OUT:.+]] = tensor.empty
// CHECK %[[RES:.+]] = linalg.generic
// CHECK: arith.index_cast
// CHECK: linalg.index 2
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK return %[[OUT]]

%gather = tensor.gather %arg1[%arg0] gather_dims([0]) : (tensor<16x4xf32>, tensor<2x1xi64>) -> tensor<2x1x4xf32>
return %gather : tensor<2x1x4xf32>
}



// CHECK-LABEL: func.func @dyn_shape
func.func @dyn_shape(%arg0: tensor<?x1xi64>, %arg1: tensor<16x4xf32>) -> tensor<?x1x4xf32> {
// CHECK: arith.constant
// CHECK: %[[OUT:.+]] = tensor.empty
// CHECK %[[RES:.+]] = linalg.generic
// CHECK: arith.index_cast
// CHECK: linalg.index 2
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK return %[[OUT]]
%gather = tensor.gather %arg1[%arg0] gather_dims([0]) : (tensor<16x4xf32>, tensor<?x1xi64>) -> tensor<?x1x4xf32>
return %gather : tensor<?x1x4xf32>
}

// CHECK-LABEL: func.func @many_dims
func.func @many_dims(%arg0: tensor<5x10x2x2x1xi64>, %arg1: tensor<16x4xf32>) -> tensor<5x10x2x2x1x4xf32> {
// CHECK: %[[OUT:.+]] = tensor.empty
// CHECK %[[RES:.+]] = linalg.generic
// CHECK: arith.index_cast
// CHECK: linalg.index 5
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK return %[[OUT]]
%gather = tensor.gather %arg1[%arg0] gather_dims([0]) : (tensor<16x4xf32>, tensor<5x10x2x2x1xi64>) -> tensor<5x10x2x2x1x4xf32>
return %gather : tensor<5x10x2x2x1x4xf32>
}

// // CHECK-LABEL: func.func @gather_1x1x1x1
func.func @gather_1x1x1x1(%arg0: tensor<1x3xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x1x1x1xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1, 2]) : (tensor<3x3x3xf32>, tensor<1x3xi64>) -> tensor<1x1x1x1xf32>
// CHECK: %[[OUT:.+]] = tensor.empty
// CHECK %[[RES:.+]] = linalg.generic
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK return %[[OUT]]
return %gather : tensor<1x1x1x1xf32>
}

// CHECK-LABEL: func @gather_1x1x1x3
func.func @gather_1x1x1x3(%arg0: tensor<1x2xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x1x1x3xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1]) : (tensor<3x3x3xf32>, tensor<1x2xi64>) -> tensor<1x1x1x3xf32>
// CHECK: %[[OUT:.+]] = tensor.empty
// CHECK %[[RES:.+]] = linalg.generic
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: linalg.index 3
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK return %[[OUT]]
return %gather : tensor<1x1x1x3xf32>
}


// tests with reduced dims

// CHECK-LABEL: func @gather_1
func.func @gather_1(%arg0: tensor<1x3xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1, 2]) : (tensor<3x3x3xf32>, tensor<1x3xi64>) -> tensor<1xf32>
// CHECK: %[[OUT:.+]] = tensor.empty
// CHECK %[[RES:.+]] = linalg.generic
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK return %[[OUT]]
return %gather : tensor<1xf32>
}

// CHECK-LABEL: func.func @gather_1x3
func.func @gather_1x3(%arg0: tensor<1x2xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3xf32> {
// CHECK: %[[OUT:.+]] = tensor.empty
// CHECK %[[RES:.+]] = linalg.generic
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: linalg.index 1
// CHECK: tensor.extract
// CHECK: linalg.yield
// CHECK return %[[OUT]]
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1]) : (tensor<3x3x3xf32>, tensor<1x2xi64>) -> tensor<1x3xf32>
return %gather : tensor<1x3xf32>
}



// CHECK-LABEL: func.func @gather_1x3x3_dim_1
func.func @gather_1x3x3_dim_1(%arg0: tensor<1x1xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3x3xf32> {
// CHECK: %[[OUT:.+]] = tensor.empty
// CHECK %[[RES:.+]] = linalg.generic
// CHECK: %[[DIM1:.+]] = arith.index_cast
// CHECK-DAG: %[[DIM0:.+]] = linalg.index 1 : index
// CHECK-DAG: %[[DIM2:.+]] = linalg.index 2 : index
// CHECK: tensor.extract %arg1[%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK: linalg.yield
// CHECK return %[[OUT]]
%gather = tensor.gather %arg1[%arg0] gather_dims([1]) : (tensor<3x3x3xf32>, tensor<1x1xi64>) -> tensor<1x3x3xf32>
return %gather : tensor<1x3x3xf32>
}

// CHECK-LABEL: func.func @gather_1x3x3_dim_2
func.func @gather_1x3x3_dim_2(%arg0: tensor<1x1xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3x3xf32> {
// CHECK: %[[OUT:.+]] = tensor.empty
// CHECK %[[RES:.+]] = linalg.generic
// CHECK: %[[DIM2:.+]] = arith.index_cast
// CHECK-DAG: %[[DIM0:.+]] = linalg.index 1 : index
// CHECK-DAG: %[[DIM1:.+]] = linalg.index 2 : index
// CHECK: tensor.extract %arg1[%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK: linalg.yield
// CHECK return %[[OUT]]
%gather = tensor.gather %arg1[%arg0] gather_dims([2]) : (tensor<3x3x3xf32>, tensor<1x1xi64>) -> tensor<1x3x3xf32>
return %gather : tensor<1x3x3xf32>
}

// CHECK-LABEL: func.func @gather_1x3_dim_1_2
func.func @gather_1x3_dim_1_2(%arg0: tensor<1x2xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3xf32> {
// CHECK: %[[OUT:.+]] = tensor.empty
// CHECK %[[RES:.+]] = linalg.generic
// CHECK: %[[DIM1:.+]] = arith.index_cast
// CHECK: %[[DIM2:.+]] = arith.index_cast
// CHECK: %[[DIM0:.+]] = linalg.index 1 : index
// CHECK: tensor.extract %arg1[%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK: linalg.yield
// CHECK return %[[OUT]]
%gather = tensor.gather %arg1[%arg0] gather_dims([1, 2]) : (tensor<3x3x3xf32>, tensor<1x2xi64>) -> tensor<1x3xf32>
return %gather : tensor<1x3xf32>
}


// CHECK-LABEL: func.func @source_1
func.func @source_1(%arg0: tensor<1x2xi64>, %arg1: tensor<1x1x1xf32>) -> tensor<1x1xf32> {
// CHECK: %[[OUT:.+]] = tensor.empty
// CHECK %[[RES:.+]] = linalg.generic
// CHECK: %[[DIM1:.+]] = arith.index_cast
// CHECK: %[[DIM2:.+]] = arith.index_cast
// CHECK: %[[DIM0:.+]] = linalg.index 1 : index
// CHECK: tensor.extract %arg1[%[[DIM0]], %[[DIM1]], %[[DIM2]]]
// CHECK: linalg.yield
// CHECK return %[[OUT]]
%gather = tensor.gather %arg1[%arg0] gather_dims([1, 2]) : (tensor<1x1x1xf32>, tensor<1x2xi64>) -> tensor<1x1xf32>
return %gather : tensor<1x1xf32>
}
