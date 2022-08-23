// RUN: iree-compile --iree-hal-target-backends=llvm-cpu \
// RUN:     --iree-flow-dispatch-use-transform-dialect=%p/matmul_dispatch_spec.mlir \
// RUN:     --iree-flow-export-benchmark-funcs %s | \
// RUN: iree-benchmark-module --device=local-task | \
// RUN: FileCheck %s

!A_size = tensor<50x100xf32>
!B_size = tensor<100x50xf32>
!C_size = tensor<50x50xf32>

// CHECK: tile_matmul_with_constant
func.func @tile_matmul_with_constant(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}
