// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-block-dynamic-dimensions{test}, cse))" --split-input-file --mlir-print-local-scope %s | FileCheck %s

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
//   CHECK-DAG:   %[[M_DYNAMIC:.+]] = arith.divui %[[M]], %[[C16]]
//       CHECK:   %[[Q_BINDING:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<4x?x16x32x128xf16>>{%[[M_DYNAMIC]]}
//       CHECK:   %[[K2_DYNAMIC:.+]] = arith.divui %[[K2]], %[[C32]]
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

func.func @basic_blocking_test(%arg0 : index) -> tensor<?xf32> {
  %0 = util.assume.int %arg0<umin = 0, umax = 1024, udiv = 16> : index
  %1 = tensor.empty(%0) : tensor<?xf32>
  return %1 : tensor<?xf32>
}
// CHECK-LABEL: func @basic_blocking_test(
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}) : tensor<?x16xf32>
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EMPTY]]
//       CHECK:   return %[[COLLAPSE]]

// -----

func.func @no_blocking(%arg0 : index) -> tensor<?xf32> {
  %1 = tensor.empty(%arg0) : tensor<?xf32>
  return %1 : tensor<?xf32>
}
// CHECK-LABEL: func @no_blocking(
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}) : tensor<?xf32>
//       CHECK:   return %[[EMPTY]]

// -----

func.func @no_unit_blocking(%arg0 : index) -> tensor<?xf32> {
  %0 = util.assume.int %arg0<umin = 0, umax = 1024, udiv = 1> : index
  %1 = tensor.empty(%0) : tensor<?xf32>
  return %1 : tensor<?xf32>
}
// CHECK-LABEL: func @no_unit_blocking(
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}) : tensor<?xf32>
//       CHECK:   return %[[EMPTY]]
