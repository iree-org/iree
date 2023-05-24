func.func @attention() -> tensor<1x4x4xf32> {
  %init = tensor.empty() : tensor<1x4x4xf32>
  %query = util.unfoldable_constant dense<1.0> : tensor<1x4x4xf32>
  %key = util.unfoldable_constant dense<0.5> : tensor<1x4x4xf32>
  %value = util.unfoldable_constant dense<2.0> : tensor<1x4x4xf32>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<1x4x4xf32>,
        tensor<1x4x4xf32>, tensor<1x4x4xf32>) outs(%init : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  return %1 : tensor<1x4x4xf32>
}

// RUN: iree-opt %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-abi-transformation-pipeline \
// RUN:   --iree-flow-transformation-pipeline \
// RUN:   --iree-stream-transformation-pipeline \
// RUN:   --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' \
// RUN:   --iree-codegen-llvmcpu-use-transform-dialect=%p/attention_codegen_spec.mlir | \
// RUN: FileCheck %s --check-prefixes=CODEGEN-DEFAULT

// CODEGEN-DEFAULT:     hal.executable.export public @attention_dispatch_0_attention_1x4x4xf32
// CODEGEN-DEFAULT:         %[[C2:.+]] = arith.constant 2 : index
// CODEGEN-DEFAULT:         %[[C1:.+]] = arith.constant 1 : index
// CODEGEN-DEFAULT:         hal.return %[[C2]], %[[C1]], %[[C1]]

// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu \
// RUN: --iree-codegen-llvmcpu-use-transform-dialect=%p/attention_codegen_spec.mlir | \
// RUN: iree-run-module --module=- --function=attention | \
// RUN: FileCheck %s --check-prefixes=EXEC

// EXEC: 1x4x4xf32={{\[}}[2 2 2 2][2 2 2 2][2 2 2 2][2 2 2 2]{{]}}
