// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-type-propagation))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @generic_op_i4() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi4>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi4>>{%d}
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi4>>{%d} -> tensor<?xi4>
  %4 = tensor.empty(%d) : tensor<?xi4>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%2 : tensor<?xi4>) outs(%4 : tensor<?xi4>) {
      ^bb0(%arg0 : i4, %arg1 : i4):
        %6 = arith.addi %arg0, %arg0 : i4
        linalg.yield %6 : i4
    } -> tensor<?xi4>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi4> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi4>>{%d}
  return
}

// CHECK-LABEL: func.func @generic_op_i4()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = iree_tensor_ext.dispatch.tensor.load %[[IN]]{{.+}} -> tensor<?xi4>
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi4>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INTENSOR]] : tensor<?xi4>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?xi4>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i4, %[[ARG1:[a-zA-Z0-9]+]]: i4)
//   CHECK-DAG:       %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG0]] : i4
//       CHECK:       linalg.yield %[[ADD]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[GENERIC]], %[[OUT]]
