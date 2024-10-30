// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-propagate-reshapes-by-expansion))" --split-input-file %s --mlir-print-local-scope | FileCheck %s

func.func @reshape_and_lowering_config(%src: tensor<3x4xf16>, %dest: tensor<12xf16>, %dest2: tensor<12xf16>) -> tensor<12xf16> {
  %collapse = tensor.collapse_shape %src [[0, 1]] : tensor<3x4xf16> into tensor<12xf16>
  %copy = linalg.copy ins(%collapse : tensor<12xf16>) outs(%dest: tensor<12xf16>) -> tensor<12xf16>
  %copy2 = linalg.copy {lowering_config = #iree_gpu.derived_thread_config} ins(%copy : tensor<12xf16>) outs(%dest2: tensor<12xf16>) -> tensor<12xf16>
  return %copy2: tensor<12xf16>
}

// CHECK-LABEL: func @reshape_and_lowering_config
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9]+]]: tensor<3x4xf16>
//       CHECK:   %[[COPY1:.+]] = linalg.generic {{.*}} ins(%[[SRC]]
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[COPY1]]
//       CHECK:   linalg.copy
//  CHECK-SAME:     lowering_config = #iree_gpu.derived_thread_config
//  CHECK-SAME:     ins(%[[COLLAPSE]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>
func.func @fold_collapse_into_loads_dynamic() -> tensor<?x32xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0}
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, %0, 32], strides = [1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0} -> tensor<2x?x32xf32>
  %3 = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  return %3 : tensor<?x32xf32>
}
// CHECK-LABEL: func @fold_collapse_into_loads_dynamic()
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%[[CONST]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x32xf32>>{%[[SHAPE]]}
//       CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[SHAPE]], 32], strides = [1, 1]
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>
func.func @fold_expand_into_loads_dynamic() -> tensor<2x?x16x32xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0}
  %2 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, %0, 32], strides = [1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0} -> tensor<2x?x32xf32>
  %3 = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%0]
  %4 = tensor.expand_shape %2 [[0], [1, 2], [3]] output_shape [2, %3, 16, 32] : tensor<2x?x32xf32> into tensor<2x?x16x32xf32>
  return %4 : tensor<2x?x16x32xf32>
}
// CHECK-LABEL: func @fold_expand_into_loads_dynamic()
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divui %[[CONST]], %[[C16]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<2x?x16x32xf32>>{%[[SHAPE]]}
//       CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [2, %[[SHAPE]], 16, 32], strides = [1, 1, 1, 1]
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<2x?x16x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !flow.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  flow.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic(
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divui %[[CONST]], %[[C2]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !flow.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}
//       CHECK:   flow.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [2, %[[SHAPE]], 32], strides = [1, 1, 1]
//  CHECK-SAME:       !flow.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}
