// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(torch-iree-to-mips-matmul))" \
// RUN:   %s | FileCheck %s

// ─────────────────────────────────────────────────────────────────────────────
// Static-shape: torch.aten.mm on f32 tensors → mips.matmul
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @mm_static
// CHECK:         torch_c.to_builtin_tensor {{.*}} -> tensor<4x8xf32>
// CHECK:         torch_c.to_builtin_tensor {{.*}} -> tensor<8x4xf32>
// CHECK:         mips.matmul {{.*}} : tensor<4x8xf32>, tensor<8x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
// CHECK-NOT:     torch.aten.mm
func.func @mm_static(%A: !torch.vtensor<[4,8],f32>,
                     %B: !torch.vtensor<[8,4],f32>)
    -> !torch.vtensor<[4,4],f32> {
  %0 = torch.aten.mm %A, %B
      : !torch.vtensor<[4,8],f32>, !torch.vtensor<[8,4],f32>
      -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-f32 (i32) should be left untouched (pattern rejects non-f32 dtypes).
// ─────────────────────────────────────────────────────────────────────────────

// CHECK-LABEL: func.func @mm_i32_unchanged
// CHECK-NOT:     mips.matmul
// CHECK:         torch.aten.mm
func.func @mm_i32_unchanged(%A: !torch.vtensor<[4,8],si32>,
                             %B: !torch.vtensor<[8,4],si32>)
    -> !torch.vtensor<[4,4],si32> {
  %0 = torch.aten.mm %A, %B
      : !torch.vtensor<[4,8],si32>, !torch.vtensor<[8,4],si32>
      -> !torch.vtensor<[4,4],si32>
  return %0 : !torch.vtensor<[4,4],si32>
}
