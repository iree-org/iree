// RUN: iree-opt --iree-llvmcpu-expand-max-f16-to-f32 %s | FileCheck %s

func.func @test_expand_f16_maxf(%arg0: tensor<4xf16>, %arg1: tensor<4xf16>) -> tensor<4xf16>{
    %1 = arith.maxf %arg0, %arg1 : tensor<4xf16>
    return %1: tensor<4xf16>
}

// CHECK-LABEL: func @test_expand_f16_maxf
// CHECK: %[[EXT1:.*]] = arith.extf %arg0 : tensor<4xf16> to tensor<4xf32> 
// CHECK: %[[EXT2:.*]] = arith.extf %arg1 : tensor<4xf16> to tensor<4xf32>
// CHECK: %[[MAX:.*]] = arith.maxf %[[EXT1]], %[[EXT2]] : tensor<4xf32>
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[MAX]] : tensor<4xf16>
// CHECK: return %[[TRUNC]] : tensor<4xf16>

// CHECK-LABEL: func @test_expand_f16_maxf
// CHECK: %[[EXT1:.*]] = arith.extf %arg0 : vector<4xf16> to vector<4xf32> 
// CHECK: %[[EXT2:.*]] = arith.extf %arg1 : vector<4xf16> to vector<4xf32>
// CHECK: %[[MAX:.*]] = arith.maxf %[[EXT1]], %[[EXT2]] : vector<4xf32>
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[MAX]] : vector<4xf16>
// CHECK: return %[[TRUNC]] : vector<4xf16>

