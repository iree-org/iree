func.func @attention() -> tensor<1x4x4xf32> {
  %init = tensor.empty() : tensor<1x4x4xf32>
  %query = util.unfoldable_constant dense<1.0> : tensor<1x4x4xf32>
  %key = util.unfoldable_constant dense<0.5> : tensor<1x4x4xf32>
  %value = util.unfoldable_constant dense<2.0> : tensor<1x4x4xf32>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<1x4x4xf32>,
        tensor<1x4x4xf32>, tensor<1x4x4xf32>) outs(%init : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %1 : tensor<1x4x4xf32>
}

// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu \
// RUN: --iree-codegen-llvmcpu-use-transform-dialect=%p/attention_codegen_spec.mlir | \
// RUN: iree-run-module --module=- --function=attention | \
// RUN: FileCheck %s --check-prefixes=EXEC

// EXEC: 1x4x4xf32={{\[}}[2 2 2 2][2 2 2 2][2 2 2 2][2 2 2 2]{{]}}
