// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-type-propagation))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @generic_op_i4() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?xi4>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<writeonly:tensor<?xi4>>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xi4>>{%d} -> tensor<?xi4>
  %4 = tensor.empty(%d) : tensor<?xi4>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%2 : tensor<?xi4>) outs(%4 : tensor<?xi4>) {
      ^bb0(%arg0 : i4, %arg1 : i4):
        %6 = arith.addi %arg0, %arg0 : i4
        linalg.yield %6 : i4
    } -> tensor<?xi4>
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi4> -> !flow.dispatch.tensor<writeonly:tensor<?xi4>>{%d}
  return
}

// CHECK-LABEL: func.func @generic_op_i4()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]{{.+}} -> tensor<?xi4>
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi4>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INTENSOR]] : tensor<?xi4>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?xi4>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i4, %[[ARG1:[a-zA-Z0-9]+]]: i4)
//   CHECK-DAG:       %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG0]] : i4
//       CHECK:       linalg.yield %[[ADD]]
//       CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[OUT]]

// -----

// This test checks that the type propagation works correctly for packed tensors.
#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#packed = #iree_encoding.packed_storage
func.func @generic_op_i1_packed() {
  %d = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : !flow.dispatch.tensor<readonly:tensor<?xi1, #packed>>{%d}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : !flow.dispatch.tensor<writeonly:tensor<?xi1, #packed>>{%d}
  %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes=[%d], strides=[1] : !flow.dispatch.tensor<readonly:tensor<?xi1, #packed>>{%d} -> tensor<?xi1, #packed>
  %4 = tensor.empty(%d) : tensor<?xi1, #packed>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%2 : tensor<?xi1, #packed>) outs(%4 : tensor<?xi1, #packed>) {
      ^bb0(%arg0 : i1, %arg1 : i1):
        %6 = arith.addi %arg0, %arg0 : i1
        linalg.yield %6 : i1
    } -> tensor<?xi1, #packed>
  flow.dispatch.tensor.store %5, %1, offsets = [0], sizes=[%d], strides=[1] : tensor<?xi1, #packed> -> !flow.dispatch.tensor<writeonly:tensor<?xi1, #packed>>{%d}
  return
}

// CHECK-LABEL: func.func @generic_op_i1_packed()
//   CHECK-DAG:   %[[IN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//   CHECK-DAG:   %[[INTENSOR:.+]] = flow.dispatch.tensor.load %[[IN]]{{.+}} -> tensor<?xi1, #iree_encoding.packed_storage>
//   CHECK-DAG:   %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi1, #iree_encoding.packed_storage>
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[INTENSOR]] : tensor<?xi1, #iree_encoding.packed_storage>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?xi1, #iree_encoding.packed_storage>)
//  CHECK-NEXT:     ^bb0(%[[ARG0:[a-zA-Z0-9]+]]: i1, %[[ARG1:[a-zA-Z0-9]+]]: i1)
//   CHECK-DAG:       %[[ADD:.+]] = arith.addi %[[ARG0]], %[[ARG0]] : i1
//       CHECK:       linalg.yield %[[ADD]]
//       CHECK:   -> tensor<?xi1, #iree_encoding.packed_storage>
//       CHECK:   flow.dispatch.tensor.store %[[GENERIC]], %[[OUT]]
