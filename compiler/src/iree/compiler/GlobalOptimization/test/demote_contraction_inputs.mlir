// RUN: iree-opt --split-input-file -iree-global-opt-demote-contraction-inputs="type=bf16 operation=matmul" %s | FileCheck %s --check-prefix=BF16-MATMUL
// RUN: iree-opt --split-input-file -iree-global-opt-demote-contraction-inputs="type=bf16 operation=conv" %s | FileCheck %s --check-prefix=BF16-CONV
// RUN: iree-opt --split-input-file -iree-global-opt-demote-contraction-inputs="type=f16 operation=matmul" %s | FileCheck %s --check-prefix=F16-MATMUL
// RUN: iree-opt --split-input-file -iree-global-opt-demote-contraction-inputs="type=f16 operation=conv" %s | FileCheck %s --check-prefix=F16-CONV

util.func public @matmul_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}

// BF16-MATMUL: @matmul_f32f32f32
// BF16-MATMUL-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// BF16-MATMUL-SAME: %[[ARG1:.+]]: tensor<250x500xf32>
// BF16-MATMUL-SAME: %[[ARG2:.+]]: tensor<100x500xf32>
// BF16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG0]] : tensor<100x250xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG1]] : tensor<250x500xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: linalg.matmul
// BF16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<100x250xbf16>, tensor<250x500xbf16>)
// BF16-MATMUL-SAME: outs(%[[ARG2]] : tensor<100x500xf32>)

// F16-MATMUL: @matmul_f32f32f32
// F16-MATMUL-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// F16-MATMUL-SAME: %[[ARG1:.+]]: tensor<250x500xf32>
// F16-MATMUL-SAME: %[[ARG2:.+]]: tensor<100x500xf32>
// F16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG0]] : tensor<100x250xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG1]] : tensor<250x500xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: linalg.matmul
// F16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<100x250xf16>, tensor<250x500xf16>)
// F16-MATMUL-SAME: outs(%[[ARG2]] : tensor<100x500xf32>)

// -----

util.func public @dynamic_matmul_f32f32f32(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}

// BF16-MATMUL: @dynamic_matmul_f32f32f32
// BF16-MATMUL-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
// BF16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG0]] : tensor<?x?xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG1]] : tensor<?x?xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: linalg.matmul
// BF16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<?x?xbf16>, tensor<?x?xbf16>)
// BF16-MATMUL-SAME: outs(%[[ARG2]] : tensor<?x?xf32>)

// F16-MATMUL: @dynamic_matmul_f32f32f32
// F16-MATMUL-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
// F16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG0]] : tensor<?x?xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG1]] : tensor<?x?xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: linalg.matmul
// F16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<?x?xf16>, tensor<?x?xf16>)
// F16-MATMUL-SAME: outs(%[[ARG2]] : tensor<?x?xf32>)

// -----

util.func public @batch_matmul_f32f32f32(%arg0 : tensor<4x100x250xf32>, %arg1 : tensor<4x250x500xf32>,
    %arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<4x100x250xf32>, tensor<4x250x500xf32>)
      outs(%arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32>
  util.return %0 : tensor<4x100x500xf32>
}

// BF16-MATMUL: @batch_matmul_f32f32f32
// BF16-MATMUL-SAME: %[[ARG0:.+]]: tensor<4x100x250xf32>
// BF16-MATMUL-SAME: %[[ARG1:.+]]: tensor<4x250x500xf32>
// BF16-MATMUL-SAME: %[[ARG2:.+]]: tensor<4x100x500xf32>
// BF16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG0]] : tensor<4x100x250xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG1]] : tensor<4x250x500xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: linalg.batch_matmul
// BF16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x100x250xbf16>, tensor<4x250x500xbf16>)
// BF16-MATMUL-SAME: outs(%[[ARG2]] : tensor<4x100x500xf32>)

// F16-MATMUL: @batch_matmul_f32f32f32
// F16-MATMUL-SAME: %[[ARG0:.+]]: tensor<4x100x250xf32>
// F16-MATMUL-SAME: %[[ARG1:.+]]: tensor<4x250x500xf32>
// F16-MATMUL-SAME: %[[ARG2:.+]]: tensor<4x100x500xf32>
// F16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG0]] : tensor<4x100x250xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG1]] : tensor<4x250x500xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: linalg.batch_matmul
// F16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x100x250xf16>, tensor<4x250x500xf16>)
// F16-MATMUL-SAME: outs(%[[ARG2]] : tensor<4x100x500xf32>)

// -----

util.func public @matvec_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250xf32>,
    %arg2 : tensor<100xf32>) -> tensor<100xf32> {
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250xf32>)
      outs(%arg2 : tensor<100xf32>) -> tensor<100xf32>
  util.return %0 : tensor<100xf32>
}

// BF16-MATMUL: @matvec_f32f32f32
// BF16-MATMUL-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// BF16-MATMUL-SAME: %[[ARG1:.+]]: tensor<250xf32>
// BF16-MATMUL-SAME: %[[ARG2:.+]]: tensor<100xf32>
// BF16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG0]] : tensor<100x250xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG1]] : tensor<250xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: linalg.matvec
// BF16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<100x250xbf16>, tensor<250xbf16>)
// BF16-MATMUL-SAME: outs(%[[ARG2]] : tensor<100xf32>)

// F16-MATMUL: @matvec_f32f32f32
// F16-MATMUL-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// F16-MATMUL-SAME: %[[ARG1:.+]]: tensor<250xf32>
// F16-MATMUL-SAME: %[[ARG2:.+]]: tensor<100xf32>
// F16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG0]] : tensor<100x250xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG1]] : tensor<250xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: linalg.matvec
// F16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<100x250xf16>, tensor<250xf16>)
// F16-MATMUL-SAME: outs(%[[ARG2]] : tensor<100xf32>)

// -----

util.func public @batch_vecmat_f32f32f32(%arg0 : tensor<4x250xf32>, %arg1 : tensor<4x250x500xf32>,
    %arg2 : tensor<4x500xf32>) -> tensor<4x500xf32> {
  %0 = linalg.batch_vecmat ins(%arg0, %arg1 : tensor<4x250xf32>, tensor<4x250x500xf32>)
      outs(%arg2 : tensor<4x500xf32>) -> tensor<4x500xf32>
  util.return %0 : tensor<4x500xf32>
}

// BF16-MATMUL: @batch_vecmat_f32f32f32
// BF16-MATMUL-SAME: %[[ARG0:.+]]: tensor<4x250xf32>
// BF16-MATMUL-SAME: %[[ARG1:.+]]: tensor<4x250x500xf32>
// BF16-MATMUL-SAME: %[[ARG2:.+]]: tensor<4x500xf32>
// BF16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG0]] : tensor<4x250xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG1]] : tensor<4x250x500xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: linalg.batch_vecmat
// BF16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x250xbf16>, tensor<4x250x500xbf16>)
// BF16-MATMUL-SAME: outs(%[[ARG2]] : tensor<4x500xf32>)

// F16-MATMUL: @batch_vecmat_f32f32f32
// F16-MATMUL-SAME: %[[ARG0:.+]]: tensor<4x250xf32>
// F16-MATMUL-SAME: %[[ARG1:.+]]: tensor<4x250x500xf32>
// F16-MATMUL-SAME: %[[ARG2:.+]]: tensor<4x500xf32>
// F16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG0]] : tensor<4x250xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG1]] : tensor<4x250x500xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: linalg.batch_vecmat
// F16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x250xf16>, tensor<4x250x500xf16>)
// F16-MATMUL-SAME: outs(%[[ARG2]] : tensor<4x500xf32>)

// -----

util.func public @nonmatch_matmul_f32f32f64(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf64>) -> tensor<100x500xf64> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf64>) -> tensor<100x500xf64>
  util.return %0 : tensor<100x500xf64>
}

// BF16-MATMUL: @nonmatch_matmul_f32f32f64
// BF16-MATMUL-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// BF16-MATMUL-SAME: %[[ARG1:.+]]: tensor<250x500xf32>
// BF16-MATMUL-SAME: %[[ARG2:.+]]: tensor<100x500xf64>
// BF16-MATMUL: linalg.matmul
// BF16-MATMUL-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<100x250xf32>, tensor<250x500xf32>)
// BF16-MATMUL-SAME: outs(%[[ARG2]] : tensor<100x500xf64>)

// F16-MATMUL: @nonmatch_matmul_f32f32f64
// F16-MATMUL-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// F16-MATMUL-SAME: %[[ARG1:.+]]: tensor<250x500xf32>
// F16-MATMUL-SAME: %[[ARG2:.+]]: tensor<100x500xf64>
// F16-MATMUL: linalg.matmul
// F16-MATMUL-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<100x250xf32>, tensor<250x500xf32>)
// F16-MATMUL-SAME: outs(%[[ARG2]] : tensor<100x500xf64>)

// -----

util.func public @batch_matmul_transpose_a_f32f32f32(%arg0 : tensor<4x250x100xf32>, %arg1 : tensor<4x250x500xf32>,
    %arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32> {
  %0 = linalg.batch_matmul
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ]
      ins(%arg0, %arg1 : tensor<4x250x100xf32>, tensor<4x250x500xf32>)
      outs(%arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32>
  util.return %0 : tensor<4x100x500xf32>
}
// BF16-MATMUL-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
// BF16-MATMUL-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// BF16-MATMUL-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// BF16-MATMUL: @batch_matmul_transpose_a_f32f32f32
// BF16-MATMUL-SAME: %[[ARG0:.+]]: tensor<4x250x100xf32>
// BF16-MATMUL-SAME: %[[ARG1:.+]]: tensor<4x250x500xf32>
// BF16-MATMUL-SAME: %[[ARG2:.+]]: tensor<4x100x500xf32>
// BF16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG0]] : tensor<4x250x100xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG1]] : tensor<4x250x500xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: linalg.batch_matmul
// BF16-MATMUL-SAME: indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
// BF16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x250x100xbf16>, tensor<4x250x500xbf16>)
// BF16-MATMUL-SAME: outs(%[[ARG2]] : tensor<4x100x500xf32>)

// F16-MATMUL-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
// F16-MATMUL-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// F16-MATMUL-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// F16-MATMUL: @batch_matmul_transpose_a_f32f32f32
// F16-MATMUL-SAME: %[[ARG0:.+]]: tensor<4x250x100xf32>
// F16-MATMUL-SAME: %[[ARG1:.+]]: tensor<4x250x500xf32>
// F16-MATMUL-SAME: %[[ARG2:.+]]: tensor<4x100x500xf32>
// F16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG0]] : tensor<4x250x100xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG1]] : tensor<4x250x500xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: linalg.batch_matmul
// F16-MATMUL-SAME: indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
// F16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x250x100xf16>, tensor<4x250x500xf16>)
// F16-MATMUL-SAME: outs(%[[ARG2]] : tensor<4x100x500xf32>)

// -----

util.func public @batch_matmul_transpose_b_f32f32f32(%arg0 : tensor<4x100x250xf32>, %arg1 : tensor<4x500x250xf32>,
    %arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32> {
  %0 = linalg.batch_matmul
      indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ]
      ins(%arg0, %arg1 : tensor<4x100x250xf32>, tensor<4x500x250xf32>)
      outs(%arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32>
  util.return %0 : tensor<4x100x500xf32>
}
// BF16-MATMUL-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// BF16-MATMUL-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// BF16-MATMUL-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// BF16-MATMUL: @batch_matmul_transpose_b_f32f32f32
// BF16-MATMUL-SAME: %[[ARG0:.+]]: tensor<4x100x250xf32>
// BF16-MATMUL-SAME: %[[ARG1:.+]]: tensor<4x500x250xf32>
// BF16-MATMUL-SAME: %[[ARG2:.+]]: tensor<4x100x500xf32>
// BF16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG0]] : tensor<4x100x250xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG1]] : tensor<4x500x250xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: linalg.batch_matmul
// BF16-MATMUL-SAME: indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
// BF16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x100x250xbf16>, tensor<4x500x250xbf16>)
// BF16-MATMUL-SAME: outs(%[[ARG2]] : tensor<4x100x500xf32>)

// F16-MATMUL-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// F16-MATMUL-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// F16-MATMUL-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// F16-MATMUL: @batch_matmul_transpose_b_f32f32f32
// F16-MATMUL-SAME: %[[ARG0:.+]]: tensor<4x100x250xf32>
// F16-MATMUL-SAME: %[[ARG1:.+]]: tensor<4x500x250xf32>
// F16-MATMUL-SAME: %[[ARG2:.+]]: tensor<4x100x500xf32>
// F16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG0]] : tensor<4x100x250xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG1]] : tensor<4x500x250xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: linalg.batch_matmul
// F16-MATMUL-SAME: indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
// F16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x100x250xf16>, tensor<4x500x250xf16>)
// F16-MATMUL-SAME: outs(%[[ARG2]] : tensor<4x100x500xf32>)

// -----

util.func public @matmul_transpose_a_f32f32f32(%arg0 : tensor<250x100xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d2, d0)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ]
      ins(%arg0, %arg1 : tensor<250x100xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}
// BF16-MATMUL-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// BF16-MATMUL-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// BF16-MATMUL-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// BF16-MATMUL: @matmul_transpose_a_f32f32f32
// BF16-MATMUL-SAME: %[[ARG0:.+]]: tensor<250x100xf32>
// BF16-MATMUL-SAME: %[[ARG1:.+]]: tensor<250x500xf32>
// BF16-MATMUL-SAME: %[[ARG2:.+]]: tensor<100x500xf32>
// BF16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG0]] : tensor<250x100xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG1]] : tensor<250x500xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: linalg.matmul
// BF16-MATMUL-SAME: indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
// BF16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<250x100xbf16>, tensor<250x500xbf16>)
// BF16-MATMUL-SAME: outs(%[[ARG2]] : tensor<100x500xf32>)

// F16-MATMUL-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// F16-MATMUL-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// F16-MATMUL-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// F16-MATMUL: @matmul_transpose_a_f32f32f32
// F16-MATMUL-SAME: %[[ARG0:.+]]: tensor<250x100xf32>
// F16-MATMUL-SAME: %[[ARG1:.+]]: tensor<250x500xf32>
// F16-MATMUL-SAME: %[[ARG2:.+]]: tensor<100x500xf32>
// F16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG0]] : tensor<250x100xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG1]] : tensor<250x500xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: linalg.matmul
// F16-MATMUL-SAME: indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
// F16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<250x100xf16>, tensor<250x500xf16>)
// F16-MATMUL-SAME: outs(%[[ARG2]] : tensor<100x500xf32>)

// -----

util.func public @matmul_transpose_b_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<500x250xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ]
      ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<500x250xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}
// BF16-MATMUL-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// BF16-MATMUL-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// BF16-MATMUL-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// BF16-MATMUL: @matmul_transpose_b_f32f32f32
// BF16-MATMUL-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// BF16-MATMUL-SAME: %[[ARG1:.+]]: tensor<500x250xf32>
// BF16-MATMUL-SAME: %[[ARG2:.+]]: tensor<100x500xf32>
// BF16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG0]] : tensor<100x250xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// BF16-MATMUL-SAME: ins(%[[ARG1]] : tensor<500x250xf32>)
// BF16-MATMUL: arith.truncf {{.*}} : f32 to bf16
// BF16-MATMUL: linalg.matmul
// BF16-MATMUL-SAME: indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
// BF16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<100x250xbf16>, tensor<500x250xbf16>)
// BF16-MATMUL-SAME: outs(%[[ARG2]] : tensor<100x500xf32>)

// F16-MATMUL-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// F16-MATMUL-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// F16-MATMUL-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// F16-MATMUL: @matmul_transpose_b_f32f32f32
// F16-MATMUL-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// F16-MATMUL-SAME: %[[ARG1:.+]]: tensor<500x250xf32>
// F16-MATMUL-SAME: %[[ARG2:.+]]: tensor<100x500xf32>
// F16-MATMUL: %[[DEMOTED0:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG0]] : tensor<100x250xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: %[[DEMOTED1:.+]] = linalg.generic
// F16-MATMUL-SAME: ins(%[[ARG1]] : tensor<500x250xf32>)
// F16-MATMUL: arith.truncf {{.*}} : f32 to f16
// F16-MATMUL: linalg.matmul
// F16-MATMUL-SAME: indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
// F16-MATMUL-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<100x250xf16>, tensor<500x250xf16>)
// F16-MATMUL-SAME: outs(%[[ARG2]] : tensor<100x500xf32>)

// -----

util.func public @conv_2d_nchw_fchw_f32f32f32(%arg0 : tensor<1x16x130x130xf32>, %arg1 : tensor<512x16x3x3xf32>,
    %arg2 : tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32> {
    %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
         ins(%arg0, %arg1 : tensor<1x16x130x130xf32>, tensor<512x16x3x3xf32>)
         outs(%arg2 : tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
  util.return %0 : tensor<1x512x128x128xf32>
}
// BF16-CONV: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// BF16-CONV-LABEL:   util.func public @conv_2d_nchw_fchw_f32f32f32(
// BF16-CONV-SAME:                                                  %[[VAL_0:.*]]: tensor<1x16x130x130xf32>,
// BF16-CONV-SAME:                                                  %[[VAL_1:.*]]: tensor<512x16x3x3xf32>,
// BF16-CONV-SAME:                                                  %[[VAL_2:.*]]: tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32> {
// BF16-CONV:           %[[VAL_3:.*]] = tensor.empty() : tensor<1x16x130x130xbf16>
// BF16-CONV:           %[[DEMOT1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]],
// BF16-CONV-SAME:          iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// BF16-CONV-SAME:          ins(%[[VAL_0]] : tensor<1x16x130x130xf32>) outs(%[[VAL_3]] : tensor<1x16x130x130xbf16>) {
// BF16-CONV:           ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: bf16):
// BF16-CONV:             %[[VAL_7:.*]] = arith.truncf %[[VAL_5]] : f32 to bf16
// BF16-CONV:             linalg.yield %[[VAL_7]] : bf16
// BF16-CONV:           } -> tensor<1x16x130x130xbf16>
// BF16-CONV:           %[[VAL_8:.*]] = tensor.empty() : tensor<512x16x3x3xbf16>
// BF16-CONV:           %[[DEMOT2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]],
// BF16-CONV-SAME:          iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// BF16-CONV-SAME:          ins(%[[VAL_1]] : tensor<512x16x3x3xf32>) outs(%[[VAL_8]] : tensor<512x16x3x3xbf16>) {
// BF16-CONV:           ^bb0(%[[VAL_10:.*]]: f32, %[[VAL_11:.*]]: bf16):
// BF16-CONV:             %[[VAL_12:.*]] = arith.truncf %[[VAL_10]] : f32 to bf16
// BF16-CONV:             linalg.yield %[[VAL_12]] : bf16
// BF16-CONV:           } -> tensor<512x16x3x3xbf16>
// BF16-CONV:           %[[VAL_13:.*]] = linalg.conv_2d_nchw_fchw ins(%[[DEMOT1]], %[[DEMOT2]] : tensor<1x16x130x130xbf16>, tensor<512x16x3x3xbf16>)
// BF16-CONV-SAME:      outs(%[[VAL_2]] : tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>

// F16-CONV: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// F16-CONV-LABEL:   util.func public @conv_2d_nchw_fchw_f32f32f32(
// F16-CONV-SAME:                                                  %[[VAL_0:.*]]: tensor<1x16x130x130xf32>,
// F16-CONV-SAME:                                                  %[[VAL_1:.*]]: tensor<512x16x3x3xf32>,
// F16-CONV-SAME:                                                  %[[VAL_2:.*]]: tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32> {
// F16-CONV:           %[[VAL_3:.*]] = tensor.empty() : tensor<1x16x130x130xf16>
// F16-CONV:           %[[DEMOT1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]],
// F16-CONV-SAME:          iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// F16-CONV-SAME:          ins(%[[VAL_0]] : tensor<1x16x130x130xf32>) outs(%[[VAL_3]] : tensor<1x16x130x130xf16>) {
// F16-CONV:           ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f16):
// F16-CONV:             %[[VAL_7:.*]] = arith.truncf %[[VAL_5]] : f32 to f16
// F16-CONV:             linalg.yield %[[VAL_7]] : f16
// F16-CONV:           } -> tensor<1x16x130x130xf16>
// F16-CONV:           %[[VAL_8:.*]] = tensor.empty() : tensor<512x16x3x3xf16>
// F16-CONV:           %[[DEMOT2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]],
// F16-CONV-SAME:          iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// F16-CONV-SAME:          ins(%[[VAL_1]] : tensor<512x16x3x3xf32>) outs(%[[VAL_8]] : tensor<512x16x3x3xf16>) {
// F16-CONV:           ^bb0(%[[VAL_10:.*]]: f32, %[[VAL_11:.*]]: f16):
// F16-CONV:             %[[VAL_12:.*]] = arith.truncf %[[VAL_10]] : f32 to f16
// F16-CONV:             linalg.yield %[[VAL_12]] : f16
// F16-CONV:           } -> tensor<512x16x3x3xf16>
// F16-CONV:           %[[VAL_13:.*]] = linalg.conv_2d_nchw_fchw ins(%[[DEMOT1]], %[[DEMOT2]] : tensor<1x16x130x130xf16>, tensor<512x16x3x3xf16>)
// F16-CONV-SAME:      outs(%[[VAL_2]] : tensor<1x512x128x128xf32>) -> tensor<1x512x128x128xf32>
