// RUN: iree-opt --split-input-file --iree-util-promote-arith-bf16-to-f32 %s | FileCheck %s

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
