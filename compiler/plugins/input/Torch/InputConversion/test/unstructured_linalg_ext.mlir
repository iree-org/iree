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
