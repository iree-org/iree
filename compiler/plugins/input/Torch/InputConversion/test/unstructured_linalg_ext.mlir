// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(torch-iree-torch-unstructured-to-linalg-ext))" %s | FileCheck %s

// CHECK-DAG:         #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:         #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @fft_rfft.with_transpose(%arg0: !torch.vtensor<[3,8,16],f32>) -> !torch.vtensor<[3,5,16],complex<f32>> {
    %int-2 = torch.constant.int -2
    %none = torch.constant.none
    %out = torch.aten.fft_rfft %arg0, %none, %int-2, %none : !torch.vtensor<[3,8,16],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[3,5,16],complex<f32>>
    return %out : !torch.vtensor<[3,5,16],complex<f32>>
}
// CHECK-LABEL:   func.func @fft_rfft.with_transpose(
// CHECK-SAME:           %arg0: !torch.vtensor<[3,8,16],f32>) -> !torch.vtensor<[3,5,16],complex<f32>> {
// CHECK-DAG:         %[[INTM1:.*]] = torch.constant.int -1
// CHECK-DAG:         %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<4xf32>
// CHECK-DAG:         %[[CST_0:.*]] = arith.constant dense<{{.*}}> : tensor<4xf32>
// CHECK-DAG:         %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:         %[[CST_1:.*]] = arith.constant dense<{{.*}}> : tensor<2xf32>
// CHECK-DAG:         %[[CST_2:.*]] = arith.constant dense<{{.*}}> : tensor<2xf32>
// CHECK-DAG:         %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:         %[[CST_3:.*]] = arith.constant dense<{{.*}}> : tensor<1xf32>
// CHECK-DAG:         %[[CST_4:.*]] = arith.constant dense<{{.*}}> : tensor<1xf32>
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:         %[[CST_5:.*]] = arith.constant dense<{{.*}}> : tensor<8xi64>
// CHECK-DAG:         %[[CST_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:         %[[INT2:.*]] = torch.constant.int 2
// CHECK-DAG:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:             %[[VAR0:.*]] = torch.aten.transpose.int %arg0, %[[INT1]], %[[INT2]] : !torch.vtensor<[3,8,16],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,16,8],f32>
// CHECK:             %[[VAR1:.*]] = torch_c.to_builtin_tensor %[[VAR0]] : !torch.vtensor<[3,16,8],f32> -> tensor<3x16x8xf32>
// CHECK-DAG:         %[[VAR2:.*]] = tensor.empty() : tensor<3x16x8xf32>
// CHECK-DAG:         %[[VAR3:.*]] = tensor.empty() : tensor<3x16x8xf32>
// CHECK:             %[[VAR4:.*]]:2 = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[CST_5]] : tensor<8xi64>) outs(%[[VAR2]], %[[VAR3]] : tensor<3x16x8xf32>, tensor<3x16x8xf32>) {
// CHECK:             ^bb0(%in: i64, %out: f32, %{{.+}}: f32):
// CHECK:               %[[VAR16:.*]] = linalg.index 0 : index
// CHECK:               %[[VAR17:.*]] = linalg.index 1 : index
// CHECK:               %[[VAR18:.*]] = arith.index_cast %in : i64 to index
// CHECK:               %[[EXTRACTED:.*]] = tensor.extract %[[VAR1]][%[[VAR16]], %[[VAR17]], %[[VAR18]]] : tensor<3x16x8xf32>
// CHECK:               linalg.yield %[[EXTRACTED]], %[[CST_6]] : f32, f32
// CHECK:             } -> (tensor<3x16x8xf32>, tensor<3x16x8xf32>)
// CHECK:             %[[VAR5:.*]]:2 = iree_linalg_ext.fft ins(%[[C1]], %[[CST_4]], %[[CST_3]] : index, tensor<1xf32>, tensor<1xf32>) outs(%[[VAR4]]#0, %[[VAR4]]#1 : tensor<3x16x8xf32>, tensor<3x16x8xf32>) : tensor<3x16x8xf32>, tensor<3x16x8xf32>
// CHECK:             %[[VAR6:.*]]:2 = iree_linalg_ext.fft ins(%[[C2]], %[[CST_2]], %[[CST_1]] : index, tensor<2xf32>, tensor<2xf32>) outs(%[[VAR5]]#0, %[[VAR5]]#1 : tensor<3x16x8xf32>, tensor<3x16x8xf32>) : tensor<3x16x8xf32>, tensor<3x16x8xf32>
// CHECK:             %[[VAR7:.*]]:2 = iree_linalg_ext.fft ins(%[[C3]], %[[CST_0]], %[[CST]] : index, tensor<4xf32>, tensor<4xf32>) outs(%[[VAR6]]#0, %[[VAR6]]#1 : tensor<3x16x8xf32>, tensor<3x16x8xf32>) : tensor<3x16x8xf32>, tensor<3x16x8xf32>
// CHECK:             %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[VAR7]]#0[0, 0, 0] [3, 16, 5] [1, 1, 1] : tensor<3x16x8xf32> to tensor<3x16x5xf32>
// CHECK:             %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[VAR7]]#1[0, 0, 0] [3, 16, 5] [1, 1, 1] : tensor<3x16x8xf32> to tensor<3x16x5xf32>
// CHECK:             %[[VAR8:.*]] = torch_c.from_builtin_tensor %[[EXTRACTED_SLICE]] : tensor<3x16x5xf32> -> !torch.vtensor<[3,16,5],f32>
// CHECK:             %[[VAR9:.*]] = torch_c.from_builtin_tensor %[[EXTRACTED_SLICE_7]] : tensor<3x16x5xf32> -> !torch.vtensor<[3,16,5],f32>
// CHECK:             %[[VAR10:.*]] = torch.aten.unsqueeze %[[VAR8]], %[[INTM1]] : !torch.vtensor<[3,16,5],f32>, !torch.int -> !torch.vtensor<[3,16,5,1],f32>
// CHECK:             %[[VAR11:.*]] = torch.aten.unsqueeze %[[VAR9]], %[[INTM1]] : !torch.vtensor<[3,16,5],f32>, !torch.int -> !torch.vtensor<[3,16,5,1],f32>
// CHECK:             %[[VAR12:.*]] = torch.prim.ListConstruct %[[VAR10]], %[[VAR11]] : (!torch.vtensor<[3,16,5,1],f32>, !torch.vtensor<[3,16,5,1],f32>) -> !torch.list<vtensor<[3,16,5,1],f32>>
// CHECK:             %[[VAR13:.*]] = torch.aten.cat %[[VAR12]], %[[INTM1]] : !torch.list<vtensor<[3,16,5,1],f32>>, !torch.int -> !torch.vtensor<[3,16,5,2],f32>
// CHECK:             %[[VAR14:.*]] = torch.aten.view_as_complex %[[VAR13]] : !torch.vtensor<[3,16,5,2],f32> -> !torch.vtensor<[3,16,5],complex<f32>>
// CHECK:             %[[VAR15:.*]] = torch.aten.transpose.int %[[VAR14]], %[[INT1]], %[[INT2]] : !torch.vtensor<[3,16,5],complex<f32>>, !torch.int, !torch.int -> !torch.vtensor<[3,5,16],complex<f32>>
// CHECK:             return %[[VAR15]] : !torch.vtensor<[3,5,16],complex<f32>>

// -----
// CHECK-DAG:         #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:         #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @fft_rfft.last(%arg0: !torch.vtensor<[3,8,16],f32>) -> !torch.vtensor<[3,8,9],complex<f32>> {
    %int-1 = torch.constant.int -1
    %none = torch.constant.none
    %out = torch.aten.fft_rfft %arg0, %none, %int-1, %none : !torch.vtensor<[3,8,16],f32>, !torch.none, !torch.int, !torch.none -> !torch.vtensor<[3,8,9],complex<f32>>
    return %out : !torch.vtensor<[3,8,9],complex<f32>>
}
// CHECK-LABEL:   func.func @fft_rfft.last(
// CHECK-SAME:           %arg0: !torch.vtensor<[3,8,16],f32>) -> !torch.vtensor<[3,8,9],complex<f32>> {
// CHECK-DAG:         %[[INTM1:.*]] = torch.constant.int -1
// CHECK-DAG:         %[[CST:.*]] = arith.constant dense<{{.*}}> : tensor<8xf32>
// CHECK-DAG:         %[[CST_0:.*]] = arith.constant dense<{{.*}}> : tensor<8xf32>
// CHECK-DAG:         %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:         %[[CST_1:.*]] = arith.constant dense<{{.*}}> : tensor<4xf32>
// CHECK-DAG:         %[[CST_2:.*]] = arith.constant dense<{{.*}}> : tensor<4xf32>
// CHECK-DAG:         %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:         %[[CST_3:.*]] = arith.constant dense<{{.*}}> : tensor<2xf32>
// CHECK-DAG:         %[[CST_4:.*]] = arith.constant dense<{{.*}}> : tensor<2xf32>
// CHECK-DAG:         %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:         %[[CST_5:.*]] = arith.constant dense<{{.*}}> : tensor<1xf32>
// CHECK-DAG:         %[[CST_6:.*]] = arith.constant dense<{{.*}}> : tensor<1xf32>
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:         %[[CST_7:.*]] = arith.constant dense<{{.*}}> : tensor<16xi64>
// CHECK-DAG:         %[[CST_8:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:             %[[VAR0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[3,8,16],f32> -> tensor<3x8x16xf32>
// CHECK-DAG:         %[[VAR1:.*]] = tensor.empty() : tensor<3x8x16xf32>
// CHECK-DAG:         %[[VAR2:.*]] = tensor.empty() : tensor<3x8x16xf32>
// CHECK:             %[[VAR3:.*]]:2 = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[CST_7]] : tensor<16xi64>) outs(%[[VAR1]], %[[VAR2]] : tensor<3x8x16xf32>, tensor<3x8x16xf32>) {
// CHECK:             ^bb0(%in: i64, %out: f32, %{{.+}}: f32):
// CHECK:               %[[VAR15:.*]] = linalg.index 0 : index
// CHECK:               %[[VAR16:.*]] = linalg.index 1 : index
// CHECK:               %[[VAR17:.*]] = arith.index_cast %in : i64 to index
// CHECK:               %[[EXTRACTED:.*]] = tensor.extract %[[VAR0]][%[[VAR15]], %[[VAR16]], %[[VAR17]]] : tensor<3x8x16xf32>
// CHECK:               linalg.yield %[[EXTRACTED]], %[[CST_8]] : f32, f32
// CHECK:             } -> (tensor<3x8x16xf32>, tensor<3x8x16xf32>)
// CHECK:             %[[VAR4:.*]]:2 = iree_linalg_ext.fft ins(%[[C1]], %[[CST_6]], %[[CST_5]] : index, tensor<1xf32>, tensor<1xf32>) outs(%[[VAR3]]#0, %[[VAR3]]#1 : tensor<3x8x16xf32>, tensor<3x8x16xf32>) : tensor<3x8x16xf32>, tensor<3x8x16xf32>
// CHECK:             %[[VAR5:.*]]:2 = iree_linalg_ext.fft ins(%[[C2]], %[[CST_4]], %[[CST_3]] : index, tensor<2xf32>, tensor<2xf32>) outs(%[[VAR4]]#0, %[[VAR4]]#1 : tensor<3x8x16xf32>, tensor<3x8x16xf32>) : tensor<3x8x16xf32>, tensor<3x8x16xf32>
// CHECK:             %[[VAR6:.*]]:2 = iree_linalg_ext.fft ins(%[[C3]], %[[CST_2]], %[[CST_1]] : index, tensor<4xf32>, tensor<4xf32>) outs(%[[VAR5]]#0, %[[VAR5]]#1 : tensor<3x8x16xf32>, tensor<3x8x16xf32>) : tensor<3x8x16xf32>, tensor<3x8x16xf32>
// CHECK:             %[[VAR7:.*]]:2 = iree_linalg_ext.fft ins(%[[C4]], %[[CST_0]], %[[CST]] : index, tensor<8xf32>, tensor<8xf32>) outs(%[[VAR6]]#0, %[[VAR6]]#1 : tensor<3x8x16xf32>, tensor<3x8x16xf32>) : tensor<3x8x16xf32>, tensor<3x8x16xf32>
// CHECK:             %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[VAR7]]#0[0, 0, 0] [3, 8, 9] [1, 1, 1] : tensor<3x8x16xf32> to tensor<3x8x9xf32>
// CHECK:             %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[VAR7]]#1[0, 0, 0] [3, 8, 9] [1, 1, 1] : tensor<3x8x16xf32> to tensor<3x8x9xf32>
// CHECK:             %[[VAR8:.*]] = torch_c.from_builtin_tensor %[[EXTRACTED_SLICE]] : tensor<3x8x9xf32> -> !torch.vtensor<[3,8,9],f32>
// CHECK:             %[[VAR9:.*]] = torch_c.from_builtin_tensor %[[EXTRACTED_SLICE_9]] : tensor<3x8x9xf32> -> !torch.vtensor<[3,8,9],f32>
// CHECK:             %[[VAR10:.*]] = torch.aten.unsqueeze %[[VAR8]], %[[INTM1]] : !torch.vtensor<[3,8,9],f32>, !torch.int -> !torch.vtensor<[3,8,9,1],f32>
// CHECK:             %[[VAR11:.*]] = torch.aten.unsqueeze %[[VAR9]], %[[INTM1]] : !torch.vtensor<[3,8,9],f32>, !torch.int -> !torch.vtensor<[3,8,9,1],f32>
// CHECK:             %[[VAR12:.*]] = torch.prim.ListConstruct %[[VAR10]], %[[VAR11]] : (!torch.vtensor<[3,8,9,1],f32>, !torch.vtensor<[3,8,9,1],f32>) -> !torch.list<vtensor<[3,8,9,1],f32>>
// CHECK:             %[[VAR13:.*]] = torch.aten.cat %[[VAR12]], %[[INTM1]] : !torch.list<vtensor<[3,8,9,1],f32>>, !torch.int -> !torch.vtensor<[3,8,9,2],f32>
// CHECK:             %[[VAR14:.*]] = torch.aten.view_as_complex %[[VAR13]] : !torch.vtensor<[3,8,9,2],f32> -> !torch.vtensor<[3,8,9],complex<f32>>
// CHECK:             return %[[VAR14]] : !torch.vtensor<[3,8,9],complex<f32>>

// -----

// Test flex_attention with both score_mod and mask_mod -> online_attention.
func.func private @sdpa_score0(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],si32>, %arg3: !torch.vtensor<[],si32>, %arg4: !torch.vtensor<[],si32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}
func.func private @sdpa_mask0(%arg0: !torch.vtensor<[],si32>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],si32>, %arg3: !torch.vtensor<[],si32>) -> !torch.vtensor<[],i1> {
  %0 = torch.aten.ge.Tensor %arg2, %arg3 : !torch.vtensor<[],si32>, !torch.vtensor<[],si32> -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}
// CHECK-LABEL: func.func @flex_attn_with_scoremod_and_maskmod
func.func @flex_attn_with_scoremod_and_maskmod(%arg0: !torch.vtensor<[2,4,8,16],f32>, %arg1: !torch.vtensor<[2,4,8,16],f32>, %arg2: !torch.vtensor<[2,4,8,16],f32>) -> !torch.vtensor<[2,4,8,16],f32> {
  %float1.0 = torch.constant.float 1.000000e+00
  %false = torch.constant.bool false
  %output, %logsumexp, %maxscore = torch.hop_flex_attention %arg0, %arg1, %arg2, %float1.0, %false, %false {score_mod_fn = @sdpa_score0, mask_mod_fn = @sdpa_mask0} : !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8,16],f32>, !torch.float, !torch.bool, !torch.bool -> !torch.vtensor<[2,4,8,16],f32>, !torch.none, !torch.none
  return %output : !torch.vtensor<[2,4,8,16],f32>
}
// Fills for acc, max, sum.
// CHECK:           linalg.fill
// CHECK:           linalg.fill
// CHECK:           linalg.fill
// OnlineAttention with inlined mask_mod and score_mod in region.
// CHECK:           iree_linalg_ext.online_attention
// CHECK-SAME:        ins({{.*}} : tensor<2x4x8x16xf32>, tensor<2x4x8x16xf32>, tensor<2x4x8x16xf32>, f32)
// CHECK-SAME:        outs({{.*}} : tensor<2x4x8x16xf32>, tensor<2x4x8xf32>, tensor<2x4x8xf32>)
// Inlined mask_mod: ge + select.
// CHECK:             torch.aten.ge.Tensor
// CHECK:             arith.select
// Inlined score_mod: tanh.
// CHECK:             torch.aten.tanh
// CHECK:             iree_linalg_ext.yield
// Normalization: (1/sum) * result.
// CHECK:           linalg.generic
// CHECK:             arith.divf
// CHECK:             arith.mulf
// CHECK:             linalg.yield

// -----

// Test flex_attention with score_mod only.
func.func private @sdpa_score1(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],si32>, %arg3: !torch.vtensor<[],si32>, %arg4: !torch.vtensor<[],si32>) -> !torch.vtensor<[],f32> {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
  return %0 : !torch.vtensor<[],f32>
}
// CHECK-LABEL: func.func @flex_attn_with_scoremod_only
func.func @flex_attn_with_scoremod_only(%arg0: !torch.vtensor<[2,4,8,16],f32>, %arg1: !torch.vtensor<[2,4,8,16],f32>, %arg2: !torch.vtensor<[2,4,8,16],f32>) -> !torch.vtensor<[2,4,8,16],f32> {
  %float1.0 = torch.constant.float 1.000000e+00
  %false = torch.constant.bool false
  %output, %logsumexp, %maxscore = torch.hop_flex_attention %arg0, %arg1, %arg2, %float1.0, %false, %false {score_mod_fn = @sdpa_score1} : !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8,16],f32>, !torch.float, !torch.bool, !torch.bool -> !torch.vtensor<[2,4,8,16],f32>, !torch.none, !torch.none
  return %output : !torch.vtensor<[2,4,8,16],f32>
}
// No mask_mod ops in region.
// CHECK:           iree_linalg_ext.online_attention
// CHECK-NOT:         arith.select
// Inlined score_mod: tanh.
// CHECK:             torch.aten.tanh
// CHECK:             iree_linalg_ext.yield

// -----

// Test flex_attention with mask_mod only.
func.func private @sdpa_mask1(%arg0: !torch.vtensor<[],si32>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],si32>, %arg3: !torch.vtensor<[],si32>) -> !torch.vtensor<[],i1> {
  %0 = torch.aten.ge.Tensor %arg2, %arg3 : !torch.vtensor<[],si32>, !torch.vtensor<[],si32> -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}
// CHECK-LABEL: func.func @flex_attn_with_maskmod_only
func.func @flex_attn_with_maskmod_only(%arg0: !torch.vtensor<[2,4,8,16],f32>, %arg1: !torch.vtensor<[2,4,8,16],f32>, %arg2: !torch.vtensor<[2,4,8,16],f32>) -> !torch.vtensor<[2,4,8,16],f32> {
  %float1.0 = torch.constant.float 1.000000e+00
  %false = torch.constant.bool false
  %output, %logsumexp, %maxscore = torch.hop_flex_attention %arg0, %arg1, %arg2, %float1.0, %false, %false {mask_mod_fn = @sdpa_mask1} : !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8,16],f32>, !torch.float, !torch.bool, !torch.bool -> !torch.vtensor<[2,4,8,16],f32>, !torch.none, !torch.none
  return %output : !torch.vtensor<[2,4,8,16],f32>
}
// OnlineAttention with inlined mask_mod, no score_mod.
// CHECK:           iree_linalg_ext.online_attention
// Inlined mask_mod: ge + select.
// CHECK:             torch.aten.ge.Tensor
// CHECK:             arith.select
// No score_mod.
// CHECK-NOT:         torch.aten.tanh
// CHECK:             iree_linalg_ext.yield

// -----

// Test flex_attention without any modifications.
// CHECK-LABEL: func.func @flex_attn_without_mods
func.func @flex_attn_without_mods(%arg0: !torch.vtensor<[2,4,8,16],f32>, %arg1: !torch.vtensor<[2,4,8,16],f32>, %arg2: !torch.vtensor<[2,4,8,16],f32>) -> !torch.vtensor<[2,4,8,16],f32> {
  %float1.0 = torch.constant.float 1.000000e+00
  %false = torch.constant.bool false
  %output, %logsumexp, %maxscore = torch.hop_flex_attention %arg0, %arg1, %arg2, %float1.0, %false, %false : !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8,16],f32>, !torch.float, !torch.bool, !torch.bool -> !torch.vtensor<[2,4,8,16],f32>, !torch.none, !torch.none
  return %output : !torch.vtensor<[2,4,8,16],f32>
}
// No mask_mod or score_mod ops — plain passthrough region.
// CHECK:           iree_linalg_ext.online_attention
// CHECK-NOT:         arith.select
// CHECK-NOT:         torch.aten.tanh
// CHECK:             iree_linalg_ext.yield
// Normalization.
// CHECK:           linalg.generic
// CHECK:             arith.divf

// -----

// Test flex_attention with return_lse=true and return_max_scores=true.
// CHECK-LABEL: func.func @flex_attn_return_lse_and_maxscores
func.func @flex_attn_return_lse_and_maxscores(%arg0: !torch.vtensor<[2,4,8,16],f32>, %arg1: !torch.vtensor<[2,4,8,16],f32>, %arg2: !torch.vtensor<[2,4,8,16],f32>) -> (!torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,8],f32>) {
  %float1.0 = torch.constant.float 1.000000e+00
  %true = torch.constant.bool true
  %output, %logsumexp, %maxscore = torch.hop_flex_attention %arg0, %arg1, %arg2, %float1.0, %true, %true : !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8,16],f32>, !torch.float, !torch.bool, !torch.bool -> !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,8],f32>
  return %output, %logsumexp, %maxscore : !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,8],f32>
}
// OnlineAttention.
// CHECK:           iree_linalg_ext.online_attention
// Normalization: (1/sum) * result.
// CHECK:           linalg.generic
// CHECK:             arith.divf
// CHECK:             arith.mulf
// Logsumexp computation: log(sum) + max.
// CHECK:           linalg.generic
// CHECK:             math.log
// CHECK:             arith.addf
// Return all three results.
// CHECK:           return {{.*}} : !torch.vtensor<[2,4,8,16],f32>, !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,4,8],f32>

// -----

// CHECK-LABEL: func.func @argmax_2d_dim1
func.func @argmax_2d_dim1(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3],si64> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int1, %false : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>
  return %0 : !torch.vtensor<[3],si64>
}
// CHECK:           torch_c.to_builtin_tensor
// CHECK:           linalg.fill
// CHECK:           linalg.fill
// CHECK:           iree_linalg_ext.arg_compare dimension(1)
// CHECK:             arith.cmpf ogt
// CHECK:             iree_linalg_ext.yield
// CHECK:           linalg.generic
// CHECK:             arith.extsi {{.*}} : i32 to i64
// CHECK:           torch_c.from_builtin_tensor

// -----

// CHECK-LABEL: func.func @argmin_2d_dim0
func.func @argmin_2d_dim0(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[4],si64> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten.argmin %arg0, %int0, %false : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[4],si64>
  return %0 : !torch.vtensor<[4],si64>
}
// CHECK:           iree_linalg_ext.arg_compare dimension(0)
// CHECK:             arith.cmpf olt
// CHECK:             iree_linalg_ext.yield

// -----

// CHECK-LABEL: func.func @argmax_dim_none
func.func @argmax_dim_none(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[],si64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %none, %false : !torch.vtensor<[3,4],f32>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}
// CHECK:           torch.aten.flatten.using_ints
// CHECK:           iree_linalg_ext.arg_compare dimension(0)
// CHECK:             arith.cmpf ogt
// CHECK:             iree_linalg_ext.yield

// -----

// CHECK-LABEL: func.func @argmax_keepdim
func.func @argmax_keepdim(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,1],si64> {
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %0 = torch.aten.argmax %arg0, %int1, %true : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,1],si64>
  return %0 : !torch.vtensor<[3,1],si64>
}
// CHECK:           iree_linalg_ext.arg_compare dimension(1)
// CHECK:           torch.aten.unsqueeze

// -----

// CHECK-LABEL: func.func @argmax_neg_dim
func.func @argmax_neg_dim(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3],si64> {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int-1, %false : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>
  return %0 : !torch.vtensor<[3],si64>
}
// Negative dim should be normalized to dim=1.
// CHECK:           iree_linalg_ext.arg_compare dimension(1)

// -----

// CHECK-LABEL: func.func @argmax_int_input
func.func @argmax_int_input(%arg0: !torch.vtensor<[5],si32>) -> !torch.vtensor<[],si64> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int0, %false : !torch.vtensor<[5],si32>, !torch.int, !torch.bool -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}
// CHECK:           iree_linalg_ext.arg_compare dimension(0)
// CHECK:             arith.cmpi sgt
// CHECK:             iree_linalg_ext.yield

// -----

// CHECK-LABEL: func.func @argmax_dim_none_keepdim
func.func @argmax_dim_none_keepdim(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[1,1],si64> {
  %none = torch.constant.none
  %true = torch.constant.bool true
  %0 = torch.aten.argmax %arg0, %none, %true : !torch.vtensor<[3,4],f32>, !torch.none, !torch.bool -> !torch.vtensor<[1,1],si64>
  return %0 : !torch.vtensor<[1,1],si64>
}
// CHECK:           torch.aten.flatten.using_ints
// CHECK:           iree_linalg_ext.arg_compare dimension(0)
// CHECK:             arith.cmpf ogt
// CHECK:             iree_linalg_ext.yield
// CHECK:           torch.aten.view

// -----

// torch-mlir lowers `onnx.ArgMax{select_last_index=1}` to a flip-then-argmax
// chain followed by `(size-1) - idx` post-processing. Detect the chain and
// fold it into a single `arg_compare` with non-strict `oge` on the un-flipped
// input.
// CHECK-LABEL: func.func @argmax_select_last_index_2d_keepdim
func.func @argmax_select_last_index_2d_keepdim(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,1],si64> {
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %int3 = torch.constant.int 3
  %int1_alpha = torch.constant.int 1
  %dims = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %flipped = torch.aten.flip %arg0, %dims : !torch.vtensor<[3,4],f32>, !torch.list<int> -> !torch.vtensor<[3,4],f32>
  %argmax = torch.aten.argmax %flipped, %int1, %true : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,1],si64>
  %sub = torch.aten.sub.Scalar %argmax, %int3, %int1_alpha : !torch.vtensor<[3,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[3,1],si64>
  %abs = torch.aten.abs %sub : !torch.vtensor<[3,1],si64> -> !torch.vtensor<[3,1],si64>
  return %abs : !torch.vtensor<[3,1],si64>
}
// CHECK-NOT:       torch.aten.flip
// CHECK:           iree_linalg_ext.arg_compare dimension(1)
// CHECK:             arith.cmpf oge
// CHECK:             iree_linalg_ext.yield
// CHECK:           torch.aten.unsqueeze
// CHECK-NOT:       torch.aten.sub.Scalar
// CHECK-NOT:       torch.aten.abs

// -----

// Argmin variant: non-strict comparator becomes `ole`.
// CHECK-LABEL: func.func @argmin_select_last_index_2d
func.func @argmin_select_last_index_2d(%arg0: !torch.vtensor<[5,3],f32>) -> !torch.vtensor<[3],si64> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %int4 = torch.constant.int 4
  %int1 = torch.constant.int 1
  %dims = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %flipped = torch.aten.flip %arg0, %dims : !torch.vtensor<[5,3],f32>, !torch.list<int> -> !torch.vtensor<[5,3],f32>
  %argmin = torch.aten.argmin %flipped, %int0, %false : !torch.vtensor<[5,3],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>
  %sub = torch.aten.sub.Scalar %argmin, %int4, %int1 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[3],si64>
  %abs = torch.aten.abs %sub : !torch.vtensor<[3],si64> -> !torch.vtensor<[3],si64>
  return %abs : !torch.vtensor<[3],si64>
}
// CHECK-NOT:       torch.aten.flip
// CHECK:           iree_linalg_ext.arg_compare dimension(0)
// CHECK:             arith.cmpf ole
// CHECK:             iree_linalg_ext.yield
// CHECK-NOT:       torch.aten.sub.Scalar
// CHECK-NOT:       torch.aten.abs

// -----

// Negative-axis flip-and-argmax: dim normalization makes the flip dim and
// argmax dim equal, so the chain still folds.
// CHECK-LABEL: func.func @argmax_select_last_index_negative_axis
func.func @argmax_select_last_index_negative_axis(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3],si64> {
  %int-1 = torch.constant.int -1
  %false = torch.constant.bool false
  %int3 = torch.constant.int 3
  %int1 = torch.constant.int 1
  %dims = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
  %flipped = torch.aten.flip %arg0, %dims : !torch.vtensor<[3,4],f32>, !torch.list<int> -> !torch.vtensor<[3,4],f32>
  %argmax = torch.aten.argmax %flipped, %int-1, %false : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>
  %sub = torch.aten.sub.Scalar %argmax, %int3, %int1 : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[3],si64>
  %abs = torch.aten.abs %sub : !torch.vtensor<[3],si64> -> !torch.vtensor<[3],si64>
  return %abs : !torch.vtensor<[3],si64>
}
// CHECK-NOT:       torch.aten.flip
// CHECK:           iree_linalg_ext.arg_compare dimension(1)
// CHECK:             arith.cmpf oge

// -----

// Negative test: flip is along a different dim than argmax. The fold must
// NOT trigger because the post-processing `(size-1) - idx` would not be a
// valid mapping. The flip stays, the comparator stays strict, and the
// sub/abs chain is preserved (to be lowered later by ConvertTorchToLinalg).
// CHECK-LABEL: func.func @argmax_flip_other_dim_not_folded
func.func @argmax_flip_other_dim_not_folded(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,1],si64> {
  %int0 = torch.constant.int 0
  %int1 = torch.constant.int 1
  %true = torch.constant.bool true
  %int3 = torch.constant.int 3
  %int1_alpha = torch.constant.int 1
  %dims = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %flipped = torch.aten.flip %arg0, %dims : !torch.vtensor<[3,4],f32>, !torch.list<int> -> !torch.vtensor<[3,4],f32>
  %argmax = torch.aten.argmax %flipped, %int1, %true : !torch.vtensor<[3,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,1],si64>
  %sub = torch.aten.sub.Scalar %argmax, %int3, %int1_alpha : !torch.vtensor<[3,1],si64>, !torch.int, !torch.int -> !torch.vtensor<[3,1],si64>
  %abs = torch.aten.abs %sub : !torch.vtensor<[3,1],si64> -> !torch.vtensor<[3,1],si64>
  return %abs : !torch.vtensor<[3,1],si64>
}
// CHECK:           torch.aten.flip
// CHECK:           iree_linalg_ext.arg_compare dimension(1)
// CHECK:             arith.cmpf ogt
// CHECK:             iree_linalg_ext.yield
// CHECK:           torch.aten.sub.Scalar
// CHECK:           torch.aten.abs

// -----

// Dynamic non-reduction axis: the conversion must materialize a `tensor.dim`
// for the dynamic dim and pass it to `tensor.empty` for the result buffers.
// CHECK-LABEL: func.func @argmax_dynamic_batch
func.func @argmax_dynamic_batch(%arg0: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?],si64> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int1, %false : !torch.vtensor<[?,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[?],si64>
  return %0 : !torch.vtensor<[?],si64>
}
// CHECK:           tensor.dim
// CHECK:           tensor.empty({{.*}}) : tensor<?xf32>
// CHECK:           tensor.empty({{.*}}) : tensor<?xi32>
// CHECK:           iree_linalg_ext.arg_compare dimension(1)
// CHECK:             arith.cmpf ogt
// CHECK:             iree_linalg_ext.yield

// -----

// Float identity must be -inf / +inf, not -FLT_MAX / +FLT_MAX. PyTorch
// treats -FLT_MAX as strictly greater than -inf, so a finite seed would
// return the wrong index for inputs like [-inf, -FLT_MAX].
// CHECK-LABEL: func.func @argmax_float_init_is_neg_inf
func.func @argmax_float_init_is_neg_inf(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],si64> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int0, %false : !torch.vtensor<[5],f32>, !torch.int, !torch.bool -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}
// CHECK:           %[[NEG_INF:.+]] = arith.constant 0xFF800000 : f32
// CHECK:           linalg.fill ins(%[[NEG_INF]] : f32)
// CHECK:           iree_linalg_ext.arg_compare dimension(0)
// CHECK:             arith.cmpf ogt

// -----

// Argmin counterpart: identity must be +inf, not +FLT_MAX.
// CHECK-LABEL: func.func @argmin_float_init_is_pos_inf
func.func @argmin_float_init_is_pos_inf(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],si64> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten.argmin %arg0, %int0, %false : !torch.vtensor<[5],f32>, !torch.int, !torch.bool -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}
// CHECK:           %[[POS_INF:.+]] = arith.constant 0x7F800000 : f32
// CHECK:           linalg.fill ins(%[[POS_INF]] : f32)
// CHECK:           iree_linalg_ext.arg_compare dimension(0)
// CHECK:             arith.cmpf olt

// -----

// Unsigned integer input: must use the unsigned `ugt` predicate, not `sgt`.
// A signed comparison would treat 0xFF as -1 and pick the wrong index.
// CHECK-LABEL: func.func @argmax_ui8_input
func.func @argmax_ui8_input(%arg0: !torch.vtensor<[5],ui8>) -> !torch.vtensor<[],si64> {
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %0 = torch.aten.argmax %arg0, %int0, %false : !torch.vtensor<[5],ui8>, !torch.int, !torch.bool -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}
// CHECK:           iree_linalg_ext.arg_compare dimension(0)
// CHECK:             arith.cmpi ugt
// CHECK:             iree_linalg_ext.yield

// -----

// Unsigned integer argmin must use `ult`, not `slt`, AND seed with the
// unsigned max (UINT32_MAX = -1 as signless i32), not the signed max.
// A signed seed of INT32_MAX would miss any input element above 2^31-1.
// CHECK-LABEL: func.func @argmin_ui32_input
func.func @argmin_ui32_input(%arg0: !torch.vtensor<[3,4],ui32>) -> !torch.vtensor<[3],si64> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %0 = torch.aten.argmin %arg0, %int1, %false : !torch.vtensor<[3,4],ui32>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>
  return %0 : !torch.vtensor<[3],si64>
}
// CHECK:           %[[UMAX:.+]] = arith.constant -1 : i32
// CHECK:           linalg.fill ins(%[[UMAX]] : i32)
// CHECK:           iree_linalg_ext.arg_compare dimension(1)
// CHECK:             arith.cmpi ult
// CHECK:             iree_linalg_ext.yield

// -----

// Unsigned integer with the select-last-index chain: the non-strict path
// must also pick the unsigned variant (`uge`).
// CHECK-LABEL: func.func @argmax_select_last_index_ui8
func.func @argmax_select_last_index_ui8(%arg0: !torch.vtensor<[3,4],ui8>) -> !torch.vtensor<[3],si64> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %int3 = torch.constant.int 3
  %int1_alpha = torch.constant.int 1
  %dims = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %flipped = torch.aten.flip %arg0, %dims : !torch.vtensor<[3,4],ui8>, !torch.list<int> -> !torch.vtensor<[3,4],ui8>
  %argmax = torch.aten.argmax %flipped, %int1, %false : !torch.vtensor<[3,4],ui8>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>
  %sub = torch.aten.sub.Scalar %argmax, %int3, %int1_alpha : !torch.vtensor<[3],si64>, !torch.int, !torch.int -> !torch.vtensor<[3],si64>
  %abs = torch.aten.abs %sub : !torch.vtensor<[3],si64> -> !torch.vtensor<[3],si64>
  return %abs : !torch.vtensor<[3],si64>
}
// CHECK-NOT:       torch.aten.flip
// CHECK:           iree_linalg_ext.arg_compare dimension(1)
// CHECK:             arith.cmpi uge
// CHECK:             iree_linalg_ext.yield
// CHECK-NOT:       torch.aten.sub.Scalar
// CHECK-NOT:       torch.aten.abs
