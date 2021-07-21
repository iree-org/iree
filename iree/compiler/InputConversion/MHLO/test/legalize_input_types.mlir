// RUN: iree-opt -split-input-file -verify-diagnostics -iree-mhlo-legalize-input-types %s | IreeFileCheck %s

// CHECK-LABEL: func @constantI64
// CHECK-SAME: () -> i32
func @constantI64() -> i64 {
  // CHECK-NEXT: constant 123 : i32
  %c123 = constant 123 : i64
  return %c123 : i64
}

// CHECK-LABEL: func @argUI64
// CHECK-SAME: (%{{.*}}: ui32) -> ui32
func @argUI64(%arg: ui64) -> ui64 {
  return %arg : ui64
}

// CHECK-LABEL: func @hloConstantI64
// CHECK-SAME: () -> tensor<1xi32>
func @hloConstantI64() -> tensor<1xi64> {
  // CHECK-NEXT: mhlo.constant dense<123> : tensor<1xi32>
  %c123 = mhlo.constant dense<123> : tensor<1xi64>
  return %c123 : tensor<1xi64>
}

// CHECK-LABEL: func @hloConstantUI64
// CHECK-SAME: () -> tensor<1xui32>
func @hloConstantUI64() -> tensor<1xui64> {
  // CHECK-NEXT: mhlo.constant dense<123> : tensor<1xui32>
  %c123 = mhlo.constant dense<123> : tensor<1xui64>
  return %c123 : tensor<1xui64>
}

// -----

// CHECK-LABEL: func @constantF64
// CHECK-SAME: () -> f32
func @constantF64() -> f64 {
  // CHECK-NEXT: constant 1.234000e+02 : f32
  %c1234 = constant 123.4 : f64
  return %c1234 : f64
}

// -----

// CHECK-LABEL: func @constantSplatTensorI64
// CHECK-SAME: () -> tensor<4xi32>
func @constantSplatTensorI64() -> tensor<4xi64> {
  // CHECK-NEXT: constant dense<123> : tensor<4xi32>
  %c123 = constant dense<123> : tensor<4xi64>
  return %c123 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @constantDenseTensorI64
// CHECK-SAME: () -> tensor<4xi32>
func @constantDenseTensorI64() -> tensor<4xi64> {
  // CHECK-NEXT: constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %c123 = constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  return %c123 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @typesI64
// CHECK-SAME: (%arg0: i32) -> i32
func @typesI64(%arg0 : i64) -> i64 {
  // CHECK-NEXT: return %arg0 : i32
  return %arg0 : i64
}

// -----

// CHECK-LABEL: func @tensorTypesI64
// CHECK-SAME: (%arg0: tensor<4x4xi32>) -> tensor<4x4xi32>
func @tensorTypesI64(%arg0 : tensor<4x4xi64>) -> tensor<4x4xi64> {
  // CHECK-NEXT: return %arg0 : tensor<4x4xi32>
  return %arg0 : tensor<4x4xi64>
}

// -----

// CHECK-LABEL: func @tensorTypesF64
// CHECK-SAME: (%arg0: tensor<4x4xf32>) -> tensor<4x4xf32>
func @tensorTypesF64(%arg0 : tensor<4x4xf64>) -> tensor<4x4xf64> {
  // CHECK-NEXT: return %arg0 : tensor<4x4xf32>
  return %arg0 : tensor<4x4xf64>
}

// -----
// expected-error@+1 {{'func' op unable to legalize type of input 0}}
func @tensorUnrankedArg(%arg0 : tensor<*xi64>) -> tensor<*xi64> {
  return %arg0 : tensor<*xi64>
}

// -----
func @tensorUnrankedValue(%arg0 : tensor<4xi64>) -> tensor<4xi64> {
  // expected-error@+1 {{'tensor.cast' op unable to legalize operation types}}
  %0 = tensor.cast %arg0 : tensor<4xi64> to tensor<*xi64>
  %1 = tensor.cast %0 : tensor<*xi64> to tensor<4xi64>
  return %1 : tensor<4xi64>
}

// -----

// CHECK-LABEL: func @compareI64
// CHECK-SAME: (%arg0: tensor<i32>, %arg1: tensor<i32>) -> (i1, tensor<i32>)
func @compareI64(%arg0 : tensor<i64>, %arg1 : tensor<i64>) -> (i1, tensor<i64>) {
  // CHECK-NEXT: %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %1 = tensor.extract %0[] : tensor<i1>
  // CHECK-NEXT: cond_br %1, ^bb1(%1, %arg0 : i1, tensor<i32>), ^bb2(%1, %arg1 : i1, tensor<i32>)
  // CHECK-NEXT: ^bb1(%2: i1, %3: tensor<i32>): // pred: ^bb0
  // CHECK-NEXT: return %2, %3 : i1, tensor<i32>
  // CHECK-NEXT: ^bb2(%4: i1, %5: tensor<i32>): // pred: ^bb0
  // CHECK-NEXT: return %4, %5 : i1, tensor<i32>
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %1 = tensor.extract %0[] : tensor<i1>
  cond_br %1, ^bb1(%1, %arg0 : i1, tensor<i64>), ^bb2(%1, %arg1 : i1, tensor<i64>)
^bb1(%2 : i1, %3 : tensor<i64>):
  return %2, %3 : i1, tensor<i64>
^bb2(%4 : i1, %5 : tensor<i64>):
  return %4, %5 : i1, tensor<i64>
}

// -----

func @tensor(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>, %C: tensor<2x4xf32>)  -> tensor<2x4xf32> {
  %E = linalg.matmul ins(%A, %B: tensor<2x3xf32>, tensor<3x4xf32>)
                    outs(%C: tensor<2x4xf32>) -> tensor<2x4xf32>
  return %E : tensor<2x4xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>

func @linalg(%A: tensor<2xf32>)  -> tensor<2xf32> {
  %init = linalg.init_tensor [2] : tensor<2xf32>
  %generic = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%A : tensor<2xf32>) outs(%init : tensor<2xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<2xf32>

  return %init : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @linalg_non_structured_op
// CHECK-SAME:    (%arg0: tensor<9xi32>) -> tensor<1x9xi32>
func @linalg_non_structured_op(%arg0: tensor<9xi64>) -> tensor<1x9xi64> {
  // CHECK:       %[[RES:.+]] = linalg.tensor_expand_shape %arg0 {{\[}}[0, 1]] : tensor<9xi32> into tensor<1x9xi32>
  // CHECK:       return %[[RES:.+]] : tensor<1x9xi32>
  %0 = linalg.tensor_expand_shape %arg0 [[0, 1]] : tensor<9xi64> into tensor<1x9xi64>
  return %0 : tensor<1x9xi64>
}
