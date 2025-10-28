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
// CHECK-DAG:         %[[CST_5:.*]] = arith.constant dense<{{.*}}> : tensor<3x16x8xf32>
// CHECK-DAG:         %[[CST_6:.*]] = arith.constant dense<{{.*}}> : tensor<8xi64>
// CHECK-DAG:         %[[INT2:.*]] = torch.constant.int 2
// CHECK-DAG:         %[[INT1:.*]] = torch.constant.int 1
// CHECK:             %[[VAR0:.*]] = torch.aten.transpose.int %arg0, %[[INT1]], %[[INT2]] : !torch.vtensor<[3,8,16],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,16,8],f32>
// CHECK:             %[[VAR1:.*]] = torch_c.to_builtin_tensor %[[VAR0]] : !torch.vtensor<[3,16,8],f32> -> tensor<3x16x8xf32>
// CHECK-DAG:         %[[VAR2:.*]] = tensor.empty() : tensor<3x16x8xf32>
// CHECK:             %[[VAR3:.*]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[CST_6]] : tensor<8xi64>) outs(%[[VAR2]] : tensor<3x16x8xf32>) {
// CHECK:             ^bb0(%in: i64, %out: f32):
// CHECK:               %[[VAR15:.*]] = linalg.index 0 : index
// CHECK:               %[[VAR16:.*]] = linalg.index 1 : index
// CHECK:               %[[VAR17:.*]] = arith.index_cast %in : i64 to index
// CHECK:               %[[EXTRACTED:.*]] = tensor.extract %[[VAR1]][%[[VAR15]], %[[VAR16]], %[[VAR17]]] : tensor<3x16x8xf32>
// CHECK:               linalg.yield %[[EXTRACTED]] : f32
// CHECK:             } -> tensor<3x16x8xf32>
// CHECK:             %[[VAR4:.*]]:2 = iree_linalg_ext.fft ins(%[[C1]], %[[CST_4]], %[[CST_3]] : index, tensor<1xf32>, tensor<1xf32>) outs(%[[VAR3]], %[[CST_5]] : tensor<3x16x8xf32>, tensor<3x16x8xf32>) : tensor<3x16x8xf32>, tensor<3x16x8xf32>
// CHECK:             %[[VAR5:.*]]:2 = iree_linalg_ext.fft ins(%[[C2]], %[[CST_2]], %[[CST_1]] : index, tensor<2xf32>, tensor<2xf32>) outs(%[[VAR4]]#0, %[[VAR4]]#1 : tensor<3x16x8xf32>, tensor<3x16x8xf32>) : tensor<3x16x8xf32>, tensor<3x16x8xf32>
// CHECK:             %[[VAR6:.*]]:2 = iree_linalg_ext.fft ins(%[[C3]], %[[CST_0]], %[[CST]] : index, tensor<4xf32>, tensor<4xf32>) outs(%[[VAR5]]#0, %[[VAR5]]#1 : tensor<3x16x8xf32>, tensor<3x16x8xf32>) : tensor<3x16x8xf32>, tensor<3x16x8xf32>
// CHECK:             %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[VAR6]]#0[0, 0, 0] [3, 16, 5] [1, 1, 1] : tensor<3x16x8xf32> to tensor<3x16x5xf32>
// CHECK:             %[[EXTRACTED_SLICE_7:.*]] = tensor.extract_slice %[[VAR6]]#1[0, 0, 0] [3, 16, 5] [1, 1, 1] : tensor<3x16x8xf32> to tensor<3x16x5xf32>
// CHECK:             %[[VAR7:.*]] = torch_c.from_builtin_tensor %[[EXTRACTED_SLICE]] : tensor<3x16x5xf32> -> !torch.vtensor<[3,16,5],f32>
// CHECK:             %[[VAR8:.*]] = torch_c.from_builtin_tensor %[[EXTRACTED_SLICE_7]] : tensor<3x16x5xf32> -> !torch.vtensor<[3,16,5],f32>
// CHECK:             %[[VAR9:.*]] = torch.aten.unsqueeze %[[VAR7]], %[[INTM1]] : !torch.vtensor<[3,16,5],f32>, !torch.int -> !torch.vtensor<[3,16,5,1],f32>
// CHECK:             %[[VAR10:.*]] = torch.aten.unsqueeze %[[VAR8]], %[[INTM1]] : !torch.vtensor<[3,16,5],f32>, !torch.int -> !torch.vtensor<[3,16,5,1],f32>
// CHECK:             %[[VAR11:.*]] = torch.prim.ListConstruct %[[VAR9]], %[[VAR10]] : (!torch.vtensor<[3,16,5,1],f32>, !torch.vtensor<[3,16,5,1],f32>) -> !torch.list<vtensor<[3,16,5,1],f32>>
// CHECK:             %[[VAR12:.*]] = torch.aten.cat %[[VAR11]], %[[INTM1]] : !torch.list<vtensor<[3,16,5,1],f32>>, !torch.int -> !torch.vtensor<[3,16,5,2],f32>
// CHECK:             %[[VAR13:.*]] = torch.aten.view_as_complex %[[VAR12]] : !torch.vtensor<[3,16,5,2],f32> -> !torch.vtensor<[3,16,5],complex<f32>>
// CHECK:             %[[VAR14:.*]] = torch.aten.transpose.int %[[VAR13]], %[[INT1]], %[[INT2]] : !torch.vtensor<[3,16,5],complex<f32>>, !torch.int, !torch.int -> !torch.vtensor<[3,5,16],complex<f32>>
// CHECK:             return %[[VAR14]] : !torch.vtensor<[3,5,16],complex<f32>>

// -----
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
// CHECK-DAG:         %[[CST_7:.*]] = arith.constant dense<{{.*}}> : tensor<3x8x16xf32>
// CHECK-DAG:         %[[CST_8:.*]] = arith.constant dense<{{.*}}> : tensor<16xi64>
// CHECK:             %[[VAR0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[3,8,16],f32> -> tensor<3x8x16xf32>
// CHECK-DAG:         %[[VAR1:.*]] = tensor.empty() : tensor<3x8x16xf32>
// CHECK:             %[[VAR2:.*]] = linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[CST_8]] : tensor<16xi64>) outs(%[[VAR1]] : tensor<3x8x16xf32>) {
// CHECK:             ^bb0(%in: i64, %out: f32):
// CHECK:               %[[VAR14:.*]] = linalg.index 0 : index
// CHECK:               %[[VAR15:.*]] = linalg.index 1 : index
// CHECK:               %[[VAR16:.*]] = arith.index_cast %in : i64 to index
// CHECK:               %[[EXTRACTED:.*]] = tensor.extract %[[VAR0]][%[[VAR14]], %[[VAR15]], %[[VAR16]]] : tensor<3x8x16xf32>
// CHECK:               linalg.yield %[[EXTRACTED]] : f32
// CHECK:             } -> tensor<3x8x16xf32>
// CHECK:             %[[VAR3:.*]]:2 = iree_linalg_ext.fft ins(%[[C1]], %[[CST_6]], %[[CST_5]] : index, tensor<1xf32>, tensor<1xf32>) outs(%[[VAR2]], %[[CST_7]] : tensor<3x8x16xf32>, tensor<3x8x16xf32>) : tensor<3x8x16xf32>, tensor<3x8x16xf32>
// CHECK:             %[[VAR4:.*]]:2 = iree_linalg_ext.fft ins(%[[C2]], %[[CST_4]], %[[CST_3]] : index, tensor<2xf32>, tensor<2xf32>) outs(%[[VAR3]]#0, %[[VAR3]]#1 : tensor<3x8x16xf32>, tensor<3x8x16xf32>) : tensor<3x8x16xf32>, tensor<3x8x16xf32>
// CHECK:             %[[VAR5:.*]]:2 = iree_linalg_ext.fft ins(%[[C3]], %[[CST_2]], %[[CST_1]] : index, tensor<4xf32>, tensor<4xf32>) outs(%[[VAR4]]#0, %[[VAR4]]#1 : tensor<3x8x16xf32>, tensor<3x8x16xf32>) : tensor<3x8x16xf32>, tensor<3x8x16xf32>
// CHECK:             %[[VAR6:.*]]:2 = iree_linalg_ext.fft ins(%[[C4]], %[[CST_0]], %[[CST]] : index, tensor<8xf32>, tensor<8xf32>) outs(%[[VAR5]]#0, %[[VAR5]]#1 : tensor<3x8x16xf32>, tensor<3x8x16xf32>) : tensor<3x8x16xf32>, tensor<3x8x16xf32>
// CHECK:             %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[VAR6]]#0[0, 0, 0] [3, 8, 9] [1, 1, 1] : tensor<3x8x16xf32> to tensor<3x8x9xf32>
// CHECK:             %[[EXTRACTED_SLICE_9:.*]] = tensor.extract_slice %[[VAR6]]#1[0, 0, 0] [3, 8, 9] [1, 1, 1] : tensor<3x8x16xf32> to tensor<3x8x9xf32>
// CHECK:             %[[VAR7:.*]] = torch_c.from_builtin_tensor %[[EXTRACTED_SLICE]] : tensor<3x8x9xf32> -> !torch.vtensor<[3,8,9],f32>
// CHECK:             %[[VAR8:.*]] = torch_c.from_builtin_tensor %[[EXTRACTED_SLICE_9]] : tensor<3x8x9xf32> -> !torch.vtensor<[3,8,9],f32>
// CHECK:             %[[VAR9:.*]] = torch.aten.unsqueeze %[[VAR7]], %[[INTM1]] : !torch.vtensor<[3,8,9],f32>, !torch.int -> !torch.vtensor<[3,8,9,1],f32>
// CHECK:             %[[VAR10:.*]] = torch.aten.unsqueeze %[[VAR8]], %[[INTM1]] : !torch.vtensor<[3,8,9],f32>, !torch.int -> !torch.vtensor<[3,8,9,1],f32>
// CHECK:             %[[VAR11:.*]] = torch.prim.ListConstruct %[[VAR9]], %[[VAR10]] : (!torch.vtensor<[3,8,9,1],f32>, !torch.vtensor<[3,8,9,1],f32>) -> !torch.list<vtensor<[3,8,9,1],f32>>
// CHECK:             %[[VAR12:.*]] = torch.aten.cat %[[VAR11]], %[[INTM1]] : !torch.list<vtensor<[3,8,9,1],f32>>, !torch.int -> !torch.vtensor<[3,8,9,2],f32>
// CHECK:             %[[VAR13:.*]] = torch.aten.view_as_complex %[[VAR12]] : !torch.vtensor<[3,8,9,2],f32> -> !torch.vtensor<[3,8,9],complex<f32>>
// CHECK:             return %[[VAR13]] : !torch.vtensor<[3,8,9],complex<f32>>

// -----

func.func @flex_attention_basic(%arg0: !torch.vtensor<[2,4,128,64],f32>, %arg1: !torch.vtensor<[2,4,128,64],f32>, %arg2: !torch.vtensor<[2,4,128,64],f32>) -> (!torch.vtensor<[2,4,128,64],f32>, !torch.none) {
  %none = torch.constant.none
  %none_0 = torch.constant.none
  %float1.000000e00 = torch.constant.float 1.000000e+00
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %list = torch.prim.ListConstruct : () -> !torch.list<vtensor>
  %output, %logsumexp = torch.aten.flex_attention %arg0, %arg1, %arg2, %none, %list, %float1.000000e00, %false, %none_0, %false {kv_block_size = 128 : i64, q_block_size = 128 : i64, score_mod_fn = @score_mod, mask_mod_fn = @mask_mod} : !torch.vtensor<[2,4,128,64],f32>, !torch.vtensor<[2,4,128,64],f32>, !torch.vtensor<[2,4,128,64],f32>, !torch.none, !torch.list<vtensor>, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[2,4,128,64],f32>, !torch.none
  return %output, %logsumexp : !torch.vtensor<[2,4,128,64],f32>, !torch.none
}

func.func private @score_mod(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],si32>, %arg3: !torch.vtensor<[],si32>, %arg4: !torch.vtensor<[],si32>) -> !torch.vtensor<[],f32> {
  return %arg0 : !torch.vtensor<[],f32>
}

func.func private @mask_mod(%arg0: !torch.vtensor<[],si32>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],si32>, %arg3: !torch.vtensor<[],si32>) -> !torch.vtensor<[],i1> {
  %0 = torch.aten.ge.Tensor %arg2, %arg3 : !torch.vtensor<[],si32>, !torch.vtensor<[],si32> -> !torch.vtensor<[],i1>
  return %0 : !torch.vtensor<[],i1>
}

// CHECK-DAG: #[[$MAP_Q:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
// CHECK-DAG: #[[$MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[$MAP_V:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
// CHECK-DAG: #[[$MAP_S:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
// CHECK-DAG: #[[$MAP_M:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
// CHECK-DAG: #[[$MAP_MAX:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP_SUM:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP_NORM_OUT:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[$MAP_NORM_SUM:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP_IDENTITY3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @flex_attention_basic(
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[2,4,128,64],f32>,
// CHECK-SAME:    %[[ARG1:.*]]: !torch.vtensor<[2,4,128,64],f32>,
// CHECK-SAME:    %[[ARG2:.*]]: !torch.vtensor<[2,4,128,64],f32>)

// Convert to builtin tensors
// CHECK:         %[[QUERY:.*]] = torch_c.to_builtin_tensor %[[ARG0]]
// CHECK:         %[[KEY:.*]] = torch_c.to_builtin_tensor %[[ARG1]]
// CHECK:         %[[VALUE:.*]] = torch_c.to_builtin_tensor %[[ARG2]]

// Create scale
// CHECK:         %[[SCALE:.*]] = arith.constant 1.250000e-01 : f32

// Create mask using linalg.generic
// CHECK:         %[[MASK_EMPTY:.*]] = tensor.empty() : tensor<2x4x128x128xf32>
// CHECK:         %[[MASK:.*]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:      ins()
// CHECK-SAME:      outs(%[[MASK_EMPTY]] : tensor<2x4x128x128xf32>)
// CHECK:         ^bb0(%{{.*}}: f32):
// CHECK:           %[[B_IDX:.*]] = linalg.index 0
// CHECK:           %[[H_IDX:.*]] = linalg.index 1
// CHECK:           %[[Q_IDX:.*]] = linalg.index 2
// CHECK:           %[[KV_IDX:.*]] = linalg.index 3
// CHECK:           %[[B_I32:.*]] = arith.index_cast %[[B_IDX]] : index to i32
// CHECK:           %[[H_I32:.*]] = arith.index_cast %[[H_IDX]] : index to i32
// CHECK:           %[[Q_I32:.*]] = arith.index_cast %[[Q_IDX]] : index to i32
// CHECK:           %[[KV_I32:.*]] = arith.index_cast %[[KV_IDX]] : index to i32
// CHECK:           %[[B_TENSOR:.*]] = tensor.from_elements %[[B_I32]] : tensor<i32>
// CHECK:           %[[H_TENSOR:.*]] = tensor.from_elements %[[H_I32]] : tensor<i32>
// CHECK:           %[[Q_TENSOR:.*]] = tensor.from_elements %[[Q_I32]] : tensor<i32>
// CHECK:           %[[KV_TENSOR:.*]] = tensor.from_elements %[[KV_I32]] : tensor<i32>
// CHECK:           %[[MASK_RESULT:.*]] = func.call @mask_mod(%[[B_TENSOR]], %[[H_TENSOR]], %[[Q_TENSOR]], %[[KV_TENSOR]])
// CHECK:           %[[MASK_BOOL:.*]] = tensor.extract %[[MASK_RESULT]][]
// CHECK:           %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[NEGINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK:           %[[MASK_VAL:.*]] = arith.select %[[MASK_BOOL]], %[[ZERO]], %[[NEGINF]] : f32
// CHECK:           linalg.yield %[[MASK_VAL]]

// Create output, max, sum tensors
// CHECK:         %[[OUTPUT_EMPTY:.*]] = tensor.empty() : tensor<2x4x128x64xf32>
// CHECK:         %[[MAX_EMPTY:.*]] = tensor.empty() : tensor<2x4x128xf32>
// CHECK:         %[[SUM_EMPTY:.*]] = tensor.empty() : tensor<2x4x128xf32>

// Create online_attention op
// CHECK:         %[[ONLINE_ATTN:.*]]:3 = iree_linalg_ext.online_attention
// CHECK-SAME:      indexing_maps = [#[[$MAP_Q]], #[[$MAP_K]], #[[$MAP_V]], #[[$MAP_S]], #[[$MAP_M]], #[[$MAP_O]], #[[$MAP_MAX]], #[[$MAP_SUM]]]
// CHECK-SAME:      ins(%[[QUERY]], %[[KEY]], %[[VALUE]], %[[SCALE]], %[[MASK]]
// CHECK-SAME:      outs(%[[OUTPUT_EMPTY]], %[[MAX_EMPTY]], %[[SUM_EMPTY]]
// CHECK:         ^bb0(%[[SCORE:.*]]: f32):
// CHECK:           %[[SCORE_B_IDX:.*]] = linalg.index 0
// CHECK:           %[[SCORE_H_IDX:.*]] = linalg.index 1
// CHECK:           %[[SCORE_Q_IDX:.*]] = linalg.index 2
// CHECK:           %[[SCORE_KV_IDX:.*]] = linalg.index 3
// CHECK:           %[[SCORE_B_I32:.*]] = arith.index_cast %[[SCORE_B_IDX]] : index to i32
// CHECK:           %[[SCORE_H_I32:.*]] = arith.index_cast %[[SCORE_H_IDX]] : index to i32
// CHECK:           %[[SCORE_Q_I32:.*]] = arith.index_cast %[[SCORE_Q_IDX]] : index to i32
// CHECK:           %[[SCORE_KV_I32:.*]] = arith.index_cast %[[SCORE_KV_IDX]] : index to i32
// CHECK:           %[[SCORE_TENSOR:.*]] = tensor.from_elements %[[SCORE]] : tensor<f32>
// CHECK:           %[[SCORE_B_TENSOR:.*]] = tensor.from_elements %[[SCORE_B_I32]] : tensor<i32>
// CHECK:           %[[SCORE_H_TENSOR:.*]] = tensor.from_elements %[[SCORE_H_I32]] : tensor<i32>
// CHECK:           %[[SCORE_Q_TENSOR:.*]] = tensor.from_elements %[[SCORE_Q_I32]] : tensor<i32>
// CHECK:           %[[SCORE_KV_TENSOR:.*]] = tensor.from_elements %[[SCORE_KV_I32]] : tensor<i32>
// CHECK:           %[[MODIFIED_SCORE_TENSOR:.*]] = func.call @score_mod(%[[SCORE_TENSOR]], %[[SCORE_B_TENSOR]], %[[SCORE_H_TENSOR]], %[[SCORE_Q_TENSOR]], %[[SCORE_KV_TENSOR]])
// CHECK:           %[[MODIFIED_SCORE:.*]] = tensor.extract %[[MODIFIED_SCORE_TENSOR]][]
// CHECK:           iree_linalg_ext.yield %[[MODIFIED_SCORE]]

// Normalize output
// CHECK:         %[[NORM_EMPTY:.*]] = tensor.empty() : tensor<2x4x128x64xf32>
// CHECK:         %[[NORMALIZED:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$MAP_NORM_OUT]], #[[$MAP_NORM_SUM]], #[[$MAP_NORM_OUT]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:      ins(%[[ONLINE_ATTN]]#0, %[[ONLINE_ATTN]]#2
// CHECK-SAME:      outs(%[[NORM_EMPTY]]
// CHECK:         ^bb0(%[[UNNORM:.*]]: f32, %[[SUM_VAL:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[NORM_VAL:.*]] = arith.divf %[[UNNORM]], %[[SUM_VAL]] : f32
// CHECK:           linalg.yield %[[NORM_VAL]]

// Convert back to torch tensor
// CHECK:         %[[TORCH_OUTPUT:.*]] = torch_c.from_builtin_tensor %[[NORMALIZED]]
// CHECK:         %[[NONE:.*]] = torch.constant.none
// CHECK:         return %[[TORCH_OUTPUT]], %[[NONE]]

// -----

func.func @flex_attention_with_lse(%arg0: !torch.vtensor<[1,2,64,32],f32>, %arg1: !torch.vtensor<[1,2,64,32],f32>, %arg2: !torch.vtensor<[1,2,64,32],f32>) -> (!torch.vtensor<[1,2,64,32],f32>, !torch.vtensor<[1,2,64],f32>) {
  %none = torch.constant.none
  %none_0 = torch.constant.none
  %float5.000000e-01 = torch.constant.float 5.000000e-01
  %false = torch.constant.bool false
  %true = torch.constant.bool true
  %list = torch.prim.ListConstruct : () -> !torch.list<vtensor>
  %output, %logsumexp = torch.aten.flex_attention %arg0, %arg1, %arg2, %none, %list, %float5.000000e-01, %false, %none_0, %true {kv_block_size = 64 : i64, q_block_size = 64 : i64} : !torch.vtensor<[1,2,64,32],f32>, !torch.vtensor<[1,2,64,32],f32>, !torch.vtensor<[1,2,64,32],f32>, !torch.none, !torch.list<vtensor>, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,2,64,32],f32>, !torch.vtensor<[1,2,64],f32>
  return %output, %logsumexp : !torch.vtensor<[1,2,64,32],f32>, !torch.vtensor<[1,2,64],f32>
}

// CHECK-LABEL: func.func @flex_attention_with_lse(
// CHECK-SAME:    %[[ARG0:.*]]: !torch.vtensor<[1,2,64,32],f32>,
// CHECK-SAME:    %[[ARG1:.*]]: !torch.vtensor<[1,2,64,32],f32>,
// CHECK-SAME:    %[[ARG2:.*]]: !torch.vtensor<[1,2,64,32],f32>)

// CHECK:         %[[QUERY:.*]] = torch_c.to_builtin_tensor %[[ARG0]]
// CHECK:         %[[KEY:.*]] = torch_c.to_builtin_tensor %[[ARG1]]
// CHECK:         %[[VALUE:.*]] = torch_c.to_builtin_tensor %[[ARG2]]
// CHECK:         %[[SCALE:.*]] = arith.constant 5.000000e-01 : f32

// CHECK:         %[[OUTPUT_EMPTY:.*]] = tensor.empty() : tensor<1x2x64x32xf32>
// CHECK:         %[[MAX_EMPTY:.*]] = tensor.empty() : tensor<1x2x64xf32>
// CHECK:         %[[SUM_EMPTY:.*]] = tensor.empty() : tensor<1x2x64xf32>

// CHECK:         %[[ONLINE_ATTN:.*]]:3 = iree_linalg_ext.online_attention
// CHECK:         ^bb0(%[[SCORE:.*]]: f32):
// CHECK:           iree_linalg_ext.yield %[[SCORE]]

// Normalize output
// CHECK:         %[[NORM_EMPTY:.*]] = tensor.empty() : tensor<1x2x64x32xf32>
// CHECK:         %[[NORMALIZED:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[ONLINE_ATTN]]#0, %[[ONLINE_ATTN]]#2

// Compute logsumexp = max + log(sum)
// CHECK:         %[[LSE_EMPTY:.*]] = tensor.empty() : tensor<1x2x64xf32>
// CHECK:         %[[LSE:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[$MAP_IDENTITY3]], #[[$MAP_IDENTITY3]], #[[$MAP_IDENTITY3]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:      ins(%[[ONLINE_ATTN]]#1, %[[ONLINE_ATTN]]#2
// CHECK-SAME:      outs(%[[LSE_EMPTY]]
// CHECK:         ^bb0(%[[MAX_VAL:.*]]: f32, %[[SUM_VAL:.*]]: f32, %{{.*}}: f32):
// CHECK:           %[[LOG_SUM:.*]] = math.log %[[SUM_VAL]] : f32
// CHECK:           %[[LSE_VAL:.*]] = arith.addf %[[MAX_VAL]], %[[LOG_SUM]] : f32
// CHECK:           linalg.yield %[[LSE_VAL]]

// CHECK:         %[[TORCH_OUTPUT:.*]] = torch_c.from_builtin_tensor %[[NORMALIZED]]
// CHECK:         %[[TORCH_LSE:.*]] = torch_c.from_builtin_tensor %[[LSE]]
// CHECK:         return %[[TORCH_OUTPUT]], %[[TORCH_LSE]]
