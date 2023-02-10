
// RUN: iree-opt %s --iree-hal-target-backends=cuda \
// RUN:   --iree-abi-transformation-pipeline \
// RUN:   --iree-flow-transformation-pipeline \
// RUN:   --iree-stream-transformation-pipeline \
// RUN:   --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
// RUN:   --iree-codegen-llvmgpu-use-transform-dialect=%p/matmul_codegen_spec.mlir | \
// RUN: FileCheck %s --check-prefixes=CODEGEN-DEFAULT

// CODEGEN-DEFAULT: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CODEGEN-DEFAULT:     hal.executable.export public @matmul_static_dispatch_0_matmul_3x3x5
// CODEGEN-DEFAULT:       ^bb0(%[[DEVICE:[a-zA-Z0-9]+]]: !hal.device, %[[ARG0:[a-zA-Z0-9]+]]: index,
// CODEGEN-DEFAULT:         %[[C1:.+]] = arith.constant 1 : index
// CODEGEN-DEFAULT:         %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
// CODEGEN-DEFAULT:         hal.return %[[D0]], %[[C1]], %[[C1]]

// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-codegen-llvmgpu-use-transform-dialect=%p/matmul_codegen_spec.mlir | \
// RUN: iree-run-module --function=matmul_static \
// RUN:   --input="3x5xf32=1" \
// RUN:   --input="5x3xf32=2" \
// RUN:   --input="3x3xf32=42" | \
// RUN: FileCheck %s --check-prefixes=EXEC

// EXEC: 3x3xf32=[52 52 52][52 52 52][52 52 52]

!A_size = tensor<1024x2048xf32>
!B_size = tensor<2048x4096xf32>
!C_size = tensor<1024x4096xf32>

func.func @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}
