// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-config-add-tuner-attributes --pass-pipeline='builtin.module(func.func(iree-codegen-gpu-generalize-named-ops),iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s

func.func @matmul(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<4x4xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<4x4xf32>, tensor<4x4xf32>)
             outs(%fill: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}


//      CHECK: linalg.fill
//      CHECK: linalg.generic
// CHECK-SAME: lowering_config = #{{.*}},
// CHECK-SAME: root_op
//      CHECK: return
