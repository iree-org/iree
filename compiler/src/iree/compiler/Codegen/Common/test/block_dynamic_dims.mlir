// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-block-dynamic-dimensions, cse))" --split-input-file --mlir-print-local-scope --iree-codegen-block-dynamic-dimensions-of-contractions %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @block_attention_dims() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 8.837890e-02 : f16
  %m_in = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %k2_in = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %0:2 = util.assume.int
      %m_in<umin = 16, umax = 4080, udiv = 16>,
      %k2_in<umin = 16, umax = 4080, udiv = 32>
    : index, index
  %m = flow.dispatch.workload.ordinal %0#0, 0 : index
  %k2 = flow.dispatch.workload.ordinal %0#1, 1 : index
  %q_in = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect")
      : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%m}
  %key_in = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect")
      : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%k2}
  %value_in = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect")
      : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%k2}
  %mask_in = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags("ReadOnly|Indirect")
      : !flow.dispatch.tensor<readonly:tensor<4x32x?x?xf16>>{%m, %k2}
  %output_in = hal.interface.binding.subspan layout(#pipeline_layout) binding(4) alignment(64) offset(%c0) flags(Indirect)
      : !flow.dispatch.tensor<writeonly:tensor<4x?x32x128xf16>>{%m}
  %q = flow.dispatch.tensor.load %q_in, offsets = [0, 0, 0, 0], sizes = [4, %m, 32, 128], strides = [1, 1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%m} -> tensor<4x?x32x128xf16>
  %key = flow.dispatch.tensor.load %key_in, offsets = [0, 0, 0, 0], sizes = [4, %k2, 32, 128], strides = [1, 1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%k2} -> tensor<4x?x32x128xf16>
  %value = flow.dispatch.tensor.load %value_in, offsets = [0, 0, 0, 0], sizes = [4, %k2, 32, 128], strides = [1, 1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<4x?x32x128xf16>>{%k2} -> tensor<4x?x32x128xf16>
  %mask = flow.dispatch.tensor.load %mask_in, offsets = [0, 0, 0, 0], sizes = [4, 32, %m, %k2], strides = [1, 1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<4x32x?x?xf16>>{%m, %k2} -> tensor<4x32x?x?xf16>
  %1 = tensor.empty(%m) : tensor<4x?x32x128xf16>
  %2 = tensor.empty(%m) : tensor<4x32x?x128xf16>
  %attn = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d1, d4)>,
                       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d1, d4)>,
                       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d1, d3)>,
                       affine_map<(d0, d1, d2, d3, d4, d5) -> ()>,
                       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>,
                       affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]}
      ins(%q, %key, %value, %cst, %mask : tensor<4x?x32x128xf16>, tensor<4x?x32x128xf16>, tensor<4x?x32x128xf16>, f16, tensor<4x32x?x?xf16>)
      outs(%2 : tensor<4x32x?x128xf16>) {
    ^bb0(%b0 : f16) :
      iree_linalg_ext.yield %b0 : f16
  }-> tensor<4x32x?x128xf16>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%attn : tensor<4x32x?x128xf16>) outs(%1 : tensor<4x?x32x128xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x?x32x128xf16>
  flow.dispatch.tensor.store %result, %output_in, offsets = [0, 0, 0, 0], sizes = [4, %m, 32, 128], strides = [1, 1, 1, 1]
      : tensor<4x?x32x128xf16> -> !flow.dispatch.tensor<writeonly:tensor<4x?x32x128xf16>>{%m}
  return
}
// CHECK-LABEL: func @block_attention_dims()
//   CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[M:.+]] = flow.dispatch.workload.ordinal %{{.+}}, 0 : index
//   CHECK-DAG:   %[[K2:.+]] = flow.dispatch.workload.ordinal %{{.+}}, 1 : index
//   CHECK-DAG:   %[[M_DYNAMIC:.+]] = arith.divsi %[[M]], %[[C16]]
//       CHECK:   %[[Q_BINDING:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<4x?x16x32x128xf16>>{%[[M_DYNAMIC]]}
//       CHECK:   %[[K2_DYNAMIC:.+]] = arith.divsi %[[K2]], %[[C32]]
//       CHECK:   %[[K_BINDING:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<4x?x32x32x128xf16>>{%[[K2_DYNAMIC]]}
//       CHECK:   %[[V_BINDING:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       binding(2)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<4x?x32x32x128xf16>>{%[[K2_DYNAMIC]]}
//       CHECK:   %[[MASK_BINDING:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       binding(3)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<4x32x?x16x?x32xf16>>{%[[M_DYNAMIC]], %[[K2_DYNAMIC]]}
//       CHECK:   %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       binding(4)
//  CHECK-SAME:       !flow.dispatch.tensor<writeonly:tensor<4x?x16x32x128xf16>>{%[[M_DYNAMIC]]}
//       CHECK:   %[[Q:.+]] = flow.dispatch.tensor.load %[[Q_BINDING]]
//  CHECK-SAME:       sizes = [4, %[[M_DYNAMIC]], 16, 32, 128]
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<4x?x16x32x128xf16>>{%[[M_DYNAMIC]]}
//       CHECK:   %[[K:.+]] = flow.dispatch.tensor.load %[[K_BINDING]]
//  CHECK-SAME:       sizes = [4, %[[K2_DYNAMIC]], 32, 32, 128]
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<4x?x32x32x128xf16>>{%[[K2_DYNAMIC]]}
//       CHECK:   %[[V:.+]] = flow.dispatch.tensor.load %[[V_BINDING]]
//  CHECK-SAME:       sizes = [4, %[[K2_DYNAMIC]], 32, 32, 128]
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<4x?x32x32x128xf16>>{%[[K2_DYNAMIC]]}
//       CHECK:   %[[MASK:.+]] = flow.dispatch.tensor.load %[[MASK_BINDING]]
//  CHECK-SAME:       sizes = [4, 32, %[[M_DYNAMIC]], 16, %[[K2_DYNAMIC]], 32]
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<4x32x?x16x?x32xf16>>{%[[M_DYNAMIC]], %[[K2_DYNAMIC]]}
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//       CHECK:       ins(%[[Q]], %[[K]], %[[V]], %{{.+}}, %[[MASK]] :
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//       CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[OUTPUT_BINDING]]

// -----

func.func @basic_blocking_test(%arg0 : index) -> tensor<?x4096xf32> {
  %0 = util.assume.int %arg0<umin = 0, umax = 1024, udiv = 16> : index
  %lhs = tensor.empty(%0) : tensor<?x2048xf32>
  %rhs = tensor.empty() : tensor<2048x4096xf32>
  %init = tensor.empty(%0) : tensor<?x4096xf32>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<?x2048xf32>, tensor<2048x4096xf32>)
      outs(%init : tensor<?x4096xf32>) -> tensor<?x4096xf32>
  return %matmul : tensor<?x4096xf32>
}
// CHECK-LABEL: func @basic_blocking_test(
//   CHECK-DAG:   %[[LHS:.+]] = tensor.empty(%{{.+}}) : tensor<?x16x2048xf32>
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?x16x4096xf32>
//       CHECK:   %[[MATMUL:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[LHS]],
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[MATMUL]]
//       CHECK:   return %[[COLLAPSE]]

// -----

func.func @no_blocking(%arg0 : index) -> tensor<?x4096xf32> {
  %lhs = tensor.empty(%arg0) : tensor<?x2048xf32>
  %rhs = tensor.empty() : tensor<2048x4096xf32>
  %init = tensor.empty(%arg0) : tensor<?x4096xf32>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<?x2048xf32>, tensor<2048x4096xf32>)
      outs(%init : tensor<?x4096xf32>) -> tensor<?x4096xf32>
  return %matmul : tensor<?x4096xf32>
}
// CHECK-LABEL: func @no_blocking(
//   CHECK-DAG:   %[[LHS:.+]] = tensor.empty(%{{.+}}) : tensor<?x2048xf32>
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?x4096xf32>
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]],
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   return %[[MATMUL]]

// -----

func.func @no_unit_blocking(%arg0 : index) -> tensor<?x4096xf32> {
  %0 = util.assume.int %arg0<umin = 0, umax = 1024, udiv = 1> : index
  %lhs = tensor.empty(%0) : tensor<?x2048xf32>
  %rhs = tensor.empty() : tensor<2048x4096xf32>
  %init = tensor.empty(%0) : tensor<?x4096xf32>
  %matmul = linalg.matmul ins(%lhs, %rhs : tensor<?x2048xf32>, tensor<2048x4096xf32>)
      outs(%init : tensor<?x4096xf32>) -> tensor<?x4096xf32>
  return %matmul : tensor<?x4096xf32>
}
// CHECK-LABEL: func @no_unit_blocking(
//   CHECK-DAG:   %[[LHS:.+]] = tensor.empty(%{{.+}}) : tensor<?x2048xf32>
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?x4096xf32>
//       CHECK:   %[[MATMUL:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]],
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   return %[[MATMUL]]


// -----

func.func @contract_op_interface_op(%rhs : tensor<2048x4096xf16>, %m : index)
    -> tensor<?x2048xf32> {
  %0 = util.assume.int %m<udiv = 16> : index
  %lhs = tensor.empty(%0) : tensor<?x4096xf16>
  %init = tensor.empty(%0) : tensor<?x2048xf32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<?x4096xf16>, tensor<2048x4096xf16>)
      outs(%init : tensor<?x2048xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %17 = arith.extf %in : f16 to f32
    %18 = arith.extf %in_0 : f16 to f32
    %19 = arith.mulf %17, %18 : f32
    %20 = arith.addf %out, %19 : f32
    linalg.yield %20 : f32
  } -> tensor<?x2048xf32>
  return %1 : tensor<?x2048xf32>
}
// CHECK-LABEL: func @contract_op_interface_op(
//   CHECK-DAG:   %[[LHS:.+]] = tensor.empty(%{{.+}}) : tensor<?x16x4096xf16>
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?x16x2048xf32>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[LHS]],
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[GENERIC]] {{\[}}[0, 1], [2]{{\]}}
//       CHECK:   return %[[COLLAPSED]]

// -----

func.func @reshape_propagation_test(%rhs : tensor<2048x4096xf16>, %m : index)
    -> tensor<?x2048xf16> {
  %cst = arith.constant 0.0 : f32
  %0 = util.assume.int %m<udiv = 16> : index
  %lhs = tensor.empty(%0) : tensor<?x4096xf16>
  %init = tensor.empty(%0) : tensor<?x2048xf32>
  %init2 = tensor.empty(%0) : tensor<?x2048xf16>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x2048xf32>) -> tensor<?x2048xf32>
  %1 = linalg.matmul_transpose_b
      ins(%lhs, %rhs : tensor<?x4096xf16>, tensor<2048x4096xf16>)
      outs(%fill : tensor<?x2048xf32>) -> tensor<?x2048xf32>
  %2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%1 : tensor<?x2048xf32>) outs(%init2 : tensor<?x2048xf16>) {
    ^bb0(%b0 : f32, %b1 : f16):
      %3 = arith.truncf %b0 : f32 to f16
      linalg.yield %3 : f16
  } -> tensor<?x2048xf16>
  return %2 : tensor<?x2048xf16>
}
// CHECK-LABEL: func @reshape_propagation_test(
//   CHECK-DAG:   %[[LHS:.+]] = tensor.empty(%{{.+}}) : tensor<?x16x4096xf16>
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?x16x2048xf32>
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[LHS]],
//  CHECK-SAME:       outs(%[[FILL]] :
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}) : tensor<?x16x2048xf16>
//       CHECK:   %[[TRUNC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[GENERIC]] :
//  CHECK-SAME:       outs(%[[EMPTY]] :
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[TRUNC]]
//       CHECK:   return %[[COLLAPSED]]

// -----

func.func @multiple_dynamic_dims(%arg0 : index, %arg1 : index) -> tensor<?x?x4096xf32> {
  %0 = util.assume.int %arg0<umin = 0, umax = 1024, udiv = 16> : index
  %lhs = tensor.empty(%arg1, %0) : tensor<?x?x2048xf32>
  %rhs = tensor.empty(%arg1) : tensor<?x2048x4096xf32>
  %init = tensor.empty(%arg1, %0) : tensor<?x?x4096xf32>
  %matmul = linalg.batch_matmul ins(%lhs, %rhs : tensor<?x?x2048xf32>, tensor<?x2048x4096xf32>)
      outs(%init : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
  return %matmul : tensor<?x?x4096xf32>
}
// CHECK-LABEL: func @multiple_dynamic_dims(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: index,
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: index)
//   CHECK-DAG:   %[[ARG0_ASSUME:.+]] = util.assume.int %[[ARG0]]
//   CHECK-DAG:   %[[RHS:.+]] = tensor.empty(%[[ARG1]]) : tensor<?x2048x4096xf32>
//   CHECK-DAG:   %[[BLOCKED_M:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 16)>()[%[[ARG0_ASSUME]]]
//   CHECK-DAG:   %[[LHS:.+]] = tensor.empty(%[[ARG1]], %[[BLOCKED_M]]) : tensor<?x?x16x2048xf32>
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%[[ARG1]], %[[BLOCKED_M]]) : tensor<?x?x16x4096xf32>
//       CHECK:   %[[MATMUL:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]]
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[MATMUL]]
//       CHECK:   return %[[COLLAPSE]]
