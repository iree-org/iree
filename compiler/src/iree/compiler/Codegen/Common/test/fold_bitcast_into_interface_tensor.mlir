// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-fold-reshape-into-interface-tensor,canonicalize))" \
// RUN:   --split-input-file %s --mlir-print-local-scope | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>
func.func @fold_load_from_bitcast_dynamic() -> tensor<?x32xf4E2M1FN> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16xi8>>{%0}
  %2 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%0, 16], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16xi8>>{%0} -> tensor<?x16xi8>
  %3 = iree_tensor_ext.bitcast %2 : tensor<?x16xi8>{%0} -> tensor<?x32xf4E2M1FN>{%0}
  return %3 : tensor<?x32xf4E2M1FN>
}
// CHECK-LABEL: func @fold_load_from_bitcast_dynamic()
//       CHECK:   %[[SHAPE:.+]] = hal.interface.constant.load
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf4E2M1FN>>{%[[SHAPE]]}
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[SHAPE]], 32], strides = [1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf4E2M1FN>>{%[[SHAPE]]}
//   CHECK-NOT:   iree_tensor_ext.bitcast
//       CHECK:   return %[[LOAD]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_store_of_bitcast_dynamic(%arg0 : tensor<?x32xf4E2M1FN>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x16xi8>>{%0}
  %2 = iree_tensor_ext.bitcast %arg0 : tensor<?x32xf4E2M1FN>{%0} -> tensor<?x16xi8>{%0}
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [%0, 16], strides = [1, 1]
      : tensor<?x16xi8> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x16xi8>>{%0}
  return
}
// CHECK-LABEL: func @fold_store_of_bitcast_dynamic(
//       CHECK:   %[[SHAPE:.+]] = hal.interface.constant.load
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf4E2M1FN>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[SHAPE]], 32], strides = [1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf4E2M1FN>>{%[[SHAPE]]}
//   CHECK-NOT:   iree_tensor_ext.bitcast
