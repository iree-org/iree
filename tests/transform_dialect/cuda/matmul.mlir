// 16n8k16_f16
!typeA = tensor<16x16xf16>
!typeB = tensor<16x8xf16>
!typeC = tensor<16x8xf16>

func.func @matmul() -> !typeC {
  %TA = linalg.init_tensor [16, 16] : !typeA
  %TB = linalg.init_tensor [16, 8] : !typeB
  %TC = linalg.init_tensor [16, 8] : !typeC
  %c2 = arith.constant 2.0 : f16
  %c3 = arith.constant 3.0 : f16
  %c0 = arith.constant 0.0 : f16
  %A = linalg.fill ins(%c2 : f16) outs(%TA : !typeA) -> !typeA
  %B = linalg.fill ins(%c3 : f16) outs(%TB : !typeB) -> !typeB
  %C = linalg.fill ins(%c0 : f16) outs(%TC : !typeC) -> !typeC
  
  %0 = linalg.matmul ins(%A, %B : !typeA, !typeB)
                     outs(%C : !typeC) -> !typeC
  return %0 : !typeC
}

// RUN: iree-opt %s --iree-hal-target-backends=cuda \
// RUN:     --iree-abi-transformation-pipeline \
// RUN:     --iree-flow-transformation-pipeline  \
// RUN:     --iree-stream-transformation-pipeline \
// RUN:     --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass))' \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%p/matmul_codegen_spec.mlir | \
// RUN: FileCheck %s --check-prefix=CHECK

//     CHECK: device_target_cuda
//     CHECK: nvgpu.mma.sync(%{{.*}}, %{{.*}}, %{{.*}}) {mmaShape = [16, 8, 16]}

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%p/matmul_codegen_spec.mlir | \
// RUN: iree-run-module --entry_function=matmul --device=cuda |\
// RUN: FileCheck %s --check-prefix=EXEC

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 16x8xf16=[96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96][96 96 96 96 96 96 96 96]
