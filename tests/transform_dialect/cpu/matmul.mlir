!A_size = tensor<3x5xf32>
!B_size = tensor<5x3xf32>
!C_size = tensor<3x3xf32>

func.func @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}

// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-opt-data-tiling=false \
// RUN:   --iree-codegen-transform-dialect-library=%p/matmul_codegen_default_spec.mlir@codegen | \
// RUN: iree-run-module --module=- --function=matmul_static \
// RUN:   --input="3x5xf32=1" \
// RUN:   --input="5x3xf32=2" \
// RUN:   --input="3x3xf32=42" | \
// RUN: FileCheck %s --check-prefixes=EXEC

// EXEC: 3x3xf32=[52 52 52][52 52 52][52 52 52]
