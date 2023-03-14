func.func @matmul(%lhs : tensor<16x16xf16>, %rhs : tensor<16x8xf16>) -> tensor<16x8xf16> {
  %c0 = arith.constant 0.0 : f16
  %0 = tensor.empty() : tensor<16x8xf16>
  %1 = linalg.fill ins(%c0 : f16) outs(%0 : tensor<16x8xf16>) -> tensor<16x8xf16>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<16x16xf16>, tensor<16x8xf16>)
      outs(%1 : tensor<16x8xf16>) -> tensor<16x8xf16>
  return %2 : tensor<16x8xf16>
}

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-hal-cuda-llvm-target-arch=sm_80 \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%p/mma_using_layout_analysis_codegen_spec.mlir | \
// RUN: iree-run-module --function=matmul --device=cuda --input="16x16xf16=1" --input="16x8xf16=2" |\
// RUN: FileCheck %s --check-prefix=EXEC

// only checking the first 6
//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 16x8xf16=[32 32 32 32 32 32
