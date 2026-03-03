// RUN: iree-opt --split-input-file --iree-mips-lower-to-func-call %s \
// RUN:   | FileCheck %s

// ─────────────────────────────────────────────────────────────────────────────
// Basic static-shape: memref-form mips.matmul → func.call @my_matmul_kernel
// ─────────────────────────────────────────────────────────────────────────────

// CHECK:       func.func private @my_matmul_kernel
// CHECK-SAME:    {llvm.bareptr = true}
//
// CHECK-LABEL: func.func @lower_mips_matmul
// CHECK-NOT:     mips.matmul
// CHECK:         memref.extract_strided_metadata
// CHECK:         call @my_matmul_kernel
module {
  func.func @lower_mips_matmul(%A: memref<4x8xf32>,
                                %B: memref<8x4xf32>,
                                %C: memref<4x4xf32>) {
    mips.matmul %A, %B, %C
        : memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>
    return
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multiple matmuls reuse the same @my_matmul_kernel declaration.
// ─────────────────────────────────────────────────────────────────────────────

// CHECK:       func.func private @my_matmul_kernel
// Check that there is exactly one declaration (not two).
// CHECK-NOT:   func.func private @my_matmul_kernel
//
// CHECK-LABEL: func.func @two_matmuls
// CHECK:         call @my_matmul_kernel
// CHECK:         call @my_matmul_kernel
module {
  func.func @two_matmuls(%A: memref<2x4xf32>,
                          %B: memref<4x2xf32>,
                          %C: memref<2x2xf32>,
                          %D: memref<2x4xf32>,
                          %E: memref<4x2xf32>,
                          %F: memref<2x2xf32>) {
    mips.matmul %A, %B, %C
        : memref<2x4xf32>, memref<4x2xf32>, memref<2x2xf32>
    mips.matmul %D, %E, %F
        : memref<2x4xf32>, memref<4x2xf32>, memref<2x2xf32>
    return
  }
}
