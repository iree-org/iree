// RUN: iree-opt --split-input-file -iree-global-opt-demote-contraction-inputs-to-bf16 %s | FileCheck %s

util.func public @matmul_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}

// CHECK: @matmul_f32f32f32
// CHECK-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<100x500xf32>
// CHECK: %[[DEMOTED0:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG0]] : tensor<100x250xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: %[[DEMOTED1:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG1]] : tensor<250x500xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<100x250xbf16>, tensor<250x500xbf16>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<100x500xf32>)

// -----

util.func public @dynamic_matmul_f32f32f32(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}

// CHECK: @dynamic_matmul_f32f32f32
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
// CHECK: %[[DEMOTED0:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG0]] : tensor<?x?xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: %[[DEMOTED1:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG1]] : tensor<?x?xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<?x?xbf16>, tensor<?x?xbf16>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<?x?xf32>)

// -----

util.func public @batch_matmul_f32f32f32(%arg0 : tensor<4x100x250xf32>, %arg1 : tensor<4x250x500xf32>,
    %arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<4x100x250xf32>, tensor<4x250x500xf32>)
      outs(%arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32>
  util.return %0 : tensor<4x100x500xf32>
}

// CHECK: @batch_matmul_f32f32f32
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x100x250xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<4x250x500xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<4x100x500xf32>
// CHECK: %[[DEMOTED0:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG0]] : tensor<4x100x250xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: %[[DEMOTED1:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG1]] : tensor<4x250x500xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: linalg.batch_matmul
// CHECK-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x100x250xbf16>, tensor<4x250x500xbf16>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<4x100x500xf32>)

// -----

util.func public @matvec_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250xf32>,
    %arg2 : tensor<100xf32>) -> tensor<100xf32> {
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250xf32>)
      outs(%arg2 : tensor<100xf32>) -> tensor<100xf32>
  util.return %0 : tensor<100xf32>
}

// CHECK: @matvec_f32f32f32
// CHECK-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<250xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<100xf32>
// CHECK: %[[DEMOTED0:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG0]] : tensor<100x250xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: %[[DEMOTED1:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG1]] : tensor<250xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: linalg.matvec
// CHECK-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<100x250xbf16>, tensor<250xbf16>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<100xf32>)

// -----

util.func public @batch_vecmat_f32f32f32(%arg0 : tensor<4x250xf32>, %arg1 : tensor<4x250x500xf32>,
    %arg2 : tensor<4x500xf32>) -> tensor<4x500xf32> {
  %0 = linalg.batch_vecmat ins(%arg0, %arg1 : tensor<4x250xf32>, tensor<4x250x500xf32>)
      outs(%arg2 : tensor<4x500xf32>) -> tensor<4x500xf32>
  util.return %0 : tensor<4x500xf32>
}

// CHECK: @batch_vecmat_f32f32f32
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x250xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<4x250x500xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<4x500xf32>
// CHECK: %[[DEMOTED0:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG0]] : tensor<4x250xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: %[[DEMOTED1:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG1]] : tensor<4x250x500xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: linalg.batch_vecmat
// CHECK-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x250xbf16>, tensor<4x250x500xbf16>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<4x500xf32>)

// -----

util.func public @nonmatch_matmul_f32f32f64(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf64>) -> tensor<100x500xf64> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf64>) -> tensor<100x500xf64>
  util.return %0 : tensor<100x500xf64>
}

// CHECK: @nonmatch_matmul_f32f32f64
// CHECK-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<100x500xf64>
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<100x250xf32>, tensor<250x500xf32>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<100x500xf64>)

// -----

util.func public @batch_matmul_transpose_a_f32f32f32(%arg0 : tensor<4x250x100xf32>, %arg1 : tensor<4x250x500xf32>,
    %arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32> {
  %0 = linalg.batch_matmul_transpose_a ins(%arg0, %arg1 : tensor<4x250x100xf32>, tensor<4x250x500xf32>)
      outs(%arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32>
  util.return %0 : tensor<4x100x500xf32>
}

// CHECK: @batch_matmul_transpose_a_f32f32f32
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x250x100xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<4x250x500xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<4x100x500xf32>
// CHECK: %[[DEMOTED0:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG0]] : tensor<4x250x100xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: %[[DEMOTED1:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG1]] : tensor<4x250x500xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: linalg.batch_matmul_transpose_a
// CHECK-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x250x100xbf16>, tensor<4x250x500xbf16>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<4x100x500xf32>)

// -----

util.func public @batch_matmul_transpose_b_f32f32f32(%arg0 : tensor<4x100x250xf32>, %arg1 : tensor<4x500x250xf32>,
    %arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32> {
  %0 = linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : tensor<4x100x250xf32>, tensor<4x500x250xf32>)
      outs(%arg2 : tensor<4x100x500xf32>) -> tensor<4x100x500xf32>
  util.return %0 : tensor<4x100x500xf32>
}

// CHECK: @batch_matmul_transpose_b_f32f32f32
// CHECK-SAME: %[[ARG0:.+]]: tensor<4x100x250xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<4x500x250xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<4x100x500xf32>
// CHECK: %[[DEMOTED0:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG0]] : tensor<4x100x250xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: %[[DEMOTED1:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG1]] : tensor<4x500x250xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: linalg.batch_matmul_transpose_b
// CHECK-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<4x100x250xbf16>, tensor<4x500x250xbf16>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<4x100x500xf32>)

// -----

util.func public @matmul_transpose_a_f32f32f32(%arg0 : tensor<250x100xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<250x100xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}

// CHECK: @matmul_transpose_a_f32f32f32
// CHECK-SAME: %[[ARG0:.+]]: tensor<250x100xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<100x500xf32>
// CHECK: %[[DEMOTED0:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG0]] : tensor<250x100xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: %[[DEMOTED1:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG1]] : tensor<250x500xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: linalg.matmul_transpose_a
// CHECK-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<250x100xbf16>, tensor<250x500xbf16>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<100x500xf32>)

// -----

util.func public @matmul_transpose_b_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<500x250xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<500x250xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}

// CHECK: @matmul_transpose_b_f32f32f32
// CHECK-SAME: %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<500x250xf32>
// CHECK-SAME: %[[ARG2:.+]]: tensor<100x500xf32>
// CHECK: %[[DEMOTED0:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG0]] : tensor<100x250xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: %[[DEMOTED1:.+]] = linalg.generic
// CHECK-SAME: ins(%[[ARG1]] : tensor<500x250xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: linalg.matmul_transpose_b
// CHECK-SAME: ins(%[[DEMOTED0]], %[[DEMOTED1]] : tensor<100x250xbf16>, tensor<500x250xbf16>)
// CHECK-SAME: outs(%[[ARG2]] : tensor<100x500xf32>)
