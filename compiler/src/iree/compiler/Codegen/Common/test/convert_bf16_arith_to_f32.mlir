// RUN: iree-opt --split-input-file --iree-convert-bf16-arith-to-f32 %s | FileCheck %s

func.func @addf_bf16(%arg0 : tensor<128xbf16>) -> tensor<128xbf16> {
  %0 = tensor.empty() : tensor<128xbf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0 : tensor<128xbf16>) outs(%0 : tensor<128xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    %5 = arith.addf %in, %in : bf16
    linalg.yield %5 : bf16
  } -> tensor<128xbf16>
  return %1 : tensor<128xbf16>
}

// CHECK-LABEL: @addf_bf16
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[IN:.+]]: bf16, %[[OUT:.+]]: bf16):
// CHECK: %[[EXT:.+]] = arith.extf %[[IN]] : bf16 to f32
// CHECK: %[[ADD:.+]] = arith.addf %[[EXT]], %[[EXT]] : f32
// CHECK: %[[TRUNC:.+]] = arith.truncf %[[ADD]] : f32 to bf16


// -----

func.func @addf_bf16(%arg0 : bf16, %arg1 : bf16) -> bf16 {
    %0 = "arith.addf"(%arg0, %arg1) : (bf16, bf16) -> bf16
    return %0 : bf16
}

// CHECK-LABEL: @addf_bf16
// CHECK-SAME: %[[ARG0:.+]]: bf16,
// CHECK-SAME: %[[ARG1:.+]]: bf16
// CHECK: %[[EXT0:.+]] = arith.extf %[[ARG0]] : bf16 to f32
// CHECK: %[[EXT1:.+]] = arith.extf %[[ARG1]] : bf16 to f32
// CHECK: %[[ADD:.+]] = arith.addf %[[EXT0]], %[[EXT1]] : f32
// CHECK: %[[TRUNC:.+]] = arith.truncf %[[ADD]] : f32 to bf16

// -----

func.func @addf_vector_bf16(%arg0 : vector<4xbf16>, %arg1 : vector<4xbf16>) -> vector<4xbf16> {
    %0 = "arith.addf"(%arg0, %arg1) : (vector<4xbf16>, vector<4xbf16>) -> vector<4xbf16>
    return %0 : vector<4xbf16>
}

// CHECK-LABEL: @addf_vector_bf16
// CHECK-SAME: %[[ARG0:.+]]: vector<4xbf16>,
// CHECK-SAME: %[[ARG1:.+]]: vector<4xbf16>
// CHECK: %[[EXT0:.+]] = arith.extf %[[ARG0]] : vector<4xbf16> to vector<4xf32>
// CHECK: %[[EXT1:.+]] = arith.extf %[[ARG1]] : vector<4xbf16> to vector<4xf32>
// CHECK: %[[ADD:.+]] = arith.addf %[[EXT0]], %[[EXT1]] : vector<4xf32>
// CHECK: %[[TRUNC:.+]] = arith.truncf %[[ADD]] : vector<4xf32> to vector<4xbf16>

// -----

func.func @bitcast_bf16(%arg0 : vector<4xbf16>, %arg1 : vector<4xbf16>) -> vector<4xbf16> {
    %0 = arith.bitcast %arg0 : vector<4xbf16> to vector<4xi16>
    %1 = arith.bitcast %arg1 : vector<4xbf16> to vector<4xi16>
    %2 = arith.xori %0, %1 : vector<4xi16>
    %3 = arith.bitcast %2 : vector<4xi16> to vector<4xbf16>
    return %3 : vector<4xbf16>
}


// CHECK-LABEL: @bitcast_bf16
// CHECK-DAG: %[[BITCAST0:.+]] = arith.bitcast %arg0 : vector<4xbf16> to vector<4xi16>
// CHECK-DAG: %[[BITCAST1:.+]] = arith.bitcast %arg1 : vector<4xbf16> to vector<4xi16>
// CHECK-DAG: %[[XOR:.+]] = arith.xori %[[BITCAST0]], %[[BITCAST1]]
// CHECK-DAG: %[[BITCAST2:.+]] = arith.bitcast %[[XOR]]
// CHECK: return %[[BITCAST2]]

// -----

func.func @truncf_vector(%arg0 : vector<4xbf16>) -> vector<4xbf16> {
  %0 = arith.constant dense<1.0> : vector<bf16>
  %1 = vector.broadcast %0 : vector<bf16> to vector<4xbf16>
  %2 = arith.addf %1, %arg0 : vector<4xbf16>
  return %2 : vector<4xbf16>
}

// CHECK-LABEL: @truncf_vector
// CHECK: %[[CST:.+]] = arith.constant dense<1.000000e+00> : vector<4xbf16>
// CHECK: %[[VAL0:.+]] = arith.extf %arg0 : vector<4xbf16> to vector<4xf32>
// CHECK: %[[VAL1:.+]] = arith.extf %[[CST]] : vector<4xbf16> to vector<4xf32>
// CHECK: %[[VAL2:.+]] = arith.addf %[[VAL1]], %[[VAL0]] : vector<4xf32>
// CHECK: %[[VAL3:.+]] = arith.truncf %[[VAL2]] : vector<4xf32> to vector<4xbf16>
// CHECK: return %[[VAL3]] : vector<4xbf16>

// -----

func.func @extf_scalar_noop(%arg0 : vector<bf16>) -> vector<bf16> {
  %0 = arith.constant dense<1.0> : vector<bf16>
  return %0 : vector<bf16>
}

// CHECK-LABEL: @extf_scalar_noop
// CHECK: %[[CST:.+]] = arith.constant dense<1.000000e+00> : vector<bf16>
// CHECK: return %[[CST]]

// -----

func.func @store_reduction_bf16(%arg0 : vector<3xbf16>, %arg1 : vector<3xbf16>, %arg2 : memref<bf16>) {
  %cst = arith.constant dense<1.000000e+00> : vector<bf16>
  %5 = vector.extractelement %cst[] : vector<bf16>
  %6 = arith.mulf %arg0, %arg1 : vector<3xbf16>
  %7 = vector.reduction <add>, %6, %5 : vector<3xbf16> into bf16
  %8 = vector.broadcast %7 : bf16 to vector<bf16>
  %9 = vector.extractelement %8[] : vector<bf16>
  memref.store %9, %arg2[] : memref<bf16>
  return
}

// CHECK-LABEL: @store_reduction_bf16
// CHECK:  %[[CST:.+]] = arith.constant dense<1.000000e+00> : vector<bf16>
// CHECK:  %[[VAL0:.+]] = arith.extf %arg0 : vector<3xbf16> to vector<3xf32>
// CHECK:  %[[VAL1:.+]] = arith.extf %arg1 : vector<3xbf16> to vector<3xf32>
// CHECK:  %[[VAL2:.+]] = vector.extractelement %[[CST]][] : vector<bf16>
// CHECK:  %[[VAL3:.+]] = arith.extf %[[VAL2]] : bf16 to f32
// CHECK:  %[[VAL4:.+]] = arith.mulf %[[VAL0]], %[[VAL1]] : vector<3xf32>
// CHECK:  %[[VAL5:.+]] = vector.reduction <add>, %[[VAL4]], %[[VAL3]] : vector<3xf32> into f32
// CHECK:  %[[VAL6:.+]] = arith.truncf %[[VAL5]] : f32 to bf16
// CHECK:  %[[VAL7:.+]] = vector.broadcast %[[VAL6]] : bf16 to vector<bf16>
// CHECK:  %[[VAL8:.+]] = vector.extractelement %[[VAL7]][] : vector<bf16>
// CHECK:  memref.store %[[VAL8]], %arg2[] : memref<bf16>

// -----

// Regression test - preserve the scalability

// CHECK-LABEL: @fma_f32_regression
func.func @fma_f32_regression(%a : vector<[32]xf32>, %b : vector<[32]xf32>, %c : vector<[32]xf32>) -> vector<[32]xf32> {
  // CHECK: vector.fma %{{.*}}, %{{.*}}, %{{.*}} : vector<[32]xf32>
  %res = vector.fma %a, %b, %c : vector<[32]xf32>
  return %res : vector<[32]xf32>
}

// -----

func.func @outerproduct_bf16(%arg0 : vector<1xbf16>, %arg1 : vector<1xbf16>, %arg2 : vector<1x1xbf16>) -> vector<1x1xbf16> {
  %0 = vector.outerproduct %arg0, %arg1, %arg2 {kind = #vector.kind<add>} : vector<1xbf16>, vector<1xbf16>
  return %0 : vector<1x1xbf16>
}

// CHECK-LABEL: func.func @outerproduct_bf16
// CHECK-DAG: %[[EXT0:.+]] = arith.extf %arg0
// CHECK-DAG: %[[EXT1:.+]] = arith.extf %arg1
// CHECK-DAG: %[[EXT2:.+]] = arith.extf %arg2
// CHECK: %[[PROD:.+]] = vector.outerproduct %[[EXT0]], %[[EXT1]], %[[EXT2]] {kind = #vector.kind<add>} : vector<1xf32>, vector<1xf32>
// CHECK: %[[TRUNC:.+]] = arith.truncf %[[PROD]] : vector<1x1xf32> to vector<1x1xbf16>
// CHECK: return %[[TRUNC]] : vector<1x1xbf16>
