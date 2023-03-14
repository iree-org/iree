func.func @matmul(%lhs : tensor<32x64xf16>, %rhs : tensor<64x128xf16>) -> tensor<32x128xf16> {
  %c0 = arith.constant 0.0 : f16
  %0 = tensor.empty() : tensor<32x128xf16>
  %1 = linalg.fill ins(%c0 : f16) outs(%0 : tensor<32x128xf16>) -> tensor<32x128xf16>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x128xf16>)
      outs(%1 : tensor<32x128xf16>) -> tensor<32x128xf16>
  return %2 : tensor<32x128xf16>
}

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%p/reduction_v2_codegen_spec.mlir | \
// RUN: iree-run-module --function=matmul --device=cuda --input="32x64xf16=1" --input="64x128xf16=2" |\
// RUN: FileCheck %s --check-prefix=EXEC

// only checking the first 6
//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 32x128xf16=128 128 128 128 128 128
