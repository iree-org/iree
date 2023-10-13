// RUN: iree-opt --iree-llvmcpu-expand-f16-op-to-f32 %s | FileCheck %s

func.func @test_expand_f16_maxf(%arg0: tensor<4xf16>, %arg1: tensor<4xf16>) -> tensor<4xf16>{
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], 
                        iterator_types = ["parallel"]} ins(%arg0: tensor<4xf16>) outs(%arg1: tensor<4xf16>) {
        ^bb0(%in: f16, %out: f16):
        %2 = arith.maximumf %in, %out : f16
        linalg.yield %2: f16
    } -> tensor<4xf16>

    return %1 : tensor<4xf16>
}

// CHECK-LABEL: func.func @test_expand_f16_maxf
// CHECK: %[[GEN:.*]] = linalg.generic
// CHECK: %[[RHSEXT:.*]] = arith.extf %in : f16 to f32 
// CHECK: %[[LHSEXT:.*]] = arith.extf %out : f16 to f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[RHSEXT]], %[[LHSEXT]] : f32
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[MAX]] : f32 to f16
// CHECK: linalg.yield %[[TRUNC:.*]] : f16
// CHECK: return %[[GEN:.*]] : tensor<4xf16>


