// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-rocm-apply-builtin-pdl-patterns{targets=gfx942 enable-specialization=true}))' \
// RUN:   --split-input-file %s | FileCheck %s

func.func @transpose_matmul_f16(%lhs : tensor<10x20xf16>, %rhs : tensor<40x20xf16>,
    %outs : tensor<10x40xf32>) -> tensor<10x40xf32> {
  %matmul = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%lhs, %rhs : tensor<10x20xf16>, tensor<40x20xf16>)
    outs(%outs : tensor<10x40xf32>) -> tensor<10x40xf32>
  return %matmul : tensor<10x40xf32>
}
func.func @transpose_matmul_f8E4M3FNUZ(%lhs : tensor<10x20xf8E4M3FNUZ>, %rhs : tensor<40x20xf8E4M3FNUZ>,
    %outs : tensor<10x40xf32>) -> tensor<10x40xf32> {
  %matmul = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%lhs, %rhs : tensor<10x20xf8E4M3FNUZ>, tensor<40x20xf8E4M3FNUZ>)
    outs(%outs : tensor<10x40xf32>) -> tensor<10x40xf32>
  return %matmul : tensor<10x40xf32>
}
func.func @normal_matmul(%lhs : tensor<10x20xf16>, %rhs : tensor<20x40xf16>,
    %outs : tensor<10x40xf32>) -> tensor<10x40xf32> {
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<10x20xf16>, tensor<20x40xf16>)
      outs(%outs : tensor<10x40xf32>) -> tensor<10x40xf32>
  return %matmul : tensor<10x40xf32>
}
func.func @transpose_matmul_f32(%lhs : tensor<10x20xf32>, %rhs : tensor<40x20xf32>,
    %outs : tensor<10x40xf32>) -> tensor<10x40xf32> {
  %matmul = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%lhs, %rhs : tensor<10x20xf32>, tensor<40x20xf32>)
    outs(%outs : tensor<10x40xf32>) -> tensor<10x40xf32>
  return %matmul : tensor<10x40xf32>
}
// CHECK-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @transpose_matmul_f16
//       CHECK:   linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
//  CHECK-SAME:     iree_codegen.specialization_ranges
//  CHECK-SAME:       [<umin = 2048, udiv = 256>, <umin = 2048, udiv = 256>, <udiv = 64>]
//  CHECK-SAME:       [<umin = 1024, udiv = 128>, <umin = 1024, udiv = 128>, <udiv = 64>]
// CHECK-LABEL: func @transpose_matmul_f8E4M3FNUZ
//       CHECK:   linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
//  CHECK-SAME:     iree_codegen.specialization_ranges
//  CHECK-SAME:       [<umin = 2048, udiv = 256>, <umin = 2048, udiv = 256>, <udiv = 128>]
//  CHECK-SAME:       [<umin = 1024, udiv = 128>, <umin = 1024, udiv = 128>, <udiv = 128>]
//       CHECK: func @normal_matmul
//   CHECK-NOT:   iree_codegen.specialization_ranges
//       CHECK: func @transpose_matmul_f32
//   CHECK-NOT:   iree_codegen.specialization_ranges
