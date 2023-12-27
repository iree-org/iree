// RUN: iree-opt --iree-llvmcpu-expand-f16-op-to-f32 --split-input-file %s | FileCheck %s

func.func @maximumf(%arg0: tensor<4xf16>, %arg1: tensor<4xf16>, %arg2: tensor<4xf16>) -> tensor<4xf16>{
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%arg0, %arg1: tensor<4xf16>, tensor<4xf16>)
    outs(%arg2: tensor<4xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %2 = arith.maximumf %in, %in_0 : f16
    linalg.yield %2: f16
  } -> tensor<4xf16>
  return %1 : tensor<4xf16>
}

// CHECK-LABEL: func.func @maximumf
// CHECK:         %[[GEN:.*]] = linalg.generic
// CHECK:           %[[LHS:.*]] = arith.extf %{{.+}} : f16 to f32
// CHECK:           %[[RHS:.*]] = arith.extf %{{.+}} : f16 to f32
// CHECK:           %[[MAX:.*]] = arith.maximumf %[[LHS]], %[[RHS]] : f32
// CHECK:           %[[TRUNC:.*]] = arith.truncf %[[MAX]] : f32 to f16
// CHECK:           linalg.yield %[[TRUNC:.*]] : f16
// CHECK:         return %[[GEN:.*]] : tensor<4xf16>

// -----

func.func @powf(%arg0: tensor<4xf16>, %arg1: tensor<4xf16>, %arg2: tensor<4xf16>) -> tensor<4xf16>{
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%arg0, %arg1: tensor<4xf16>, tensor<4xf16>)
    outs(%arg2: tensor<4xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %2 = math.powf %in, %in_0 : f16
    linalg.yield %2: f16
  } -> tensor<4xf16>
  return %1 : tensor<4xf16>
}
// CHECK-LABEL: func.func @powf
// CHECK:         %[[GEN:.*]] = linalg.generic
// CHECK:           %[[LHS:.*]] = arith.extf %{{.+}} : f16 to f32
// CHECK:           %[[RHS:.*]] = arith.extf %{{.+}} : f16 to f32
// CHECK:           %[[POWF:.*]] = math.powf %[[LHS]], %[[RHS]] : f32
// CHECK:           %[[TRUNC:.*]] = arith.truncf %[[POWF]] : f32 to f16
// CHECK:           linalg.yield %[[TRUNC:.*]] : f16
// CHECK:         return %[[GEN:.*]] : tensor<4xf16>

