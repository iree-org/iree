// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-cleanup-buffer-alloc-view))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fold_reshape_load() {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<3x3x1x96xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<3x3x96xf32, #hal.descriptor_type<storage_buffer>>
  %2 = iree_codegen.load_from_buffer %0 : memref<3x3x1x96xf32, #hal.descriptor_type<storage_buffer>> -> tensor<3x3x1x96xf32>
  %collapsed = tensor.collapse_shape %2 [[0, 1, 2, 3]] : tensor<3x3x1x96xf32> into tensor<864xf32>
  %expanded = tensor.expand_shape %collapsed [[0, 1, 2]] output_shape [3, 3, 96] : tensor<864xf32> into tensor<3x3x96xf32>
  %barrier = util.optimization_barrier %expanded : tensor<3x3x96xf32>
  iree_codegen.store_to_buffer %barrier, %1 : tensor<3x3x96xf32> into memref<3x3x96xf32, #hal.descriptor_type<storage_buffer>>
  return
}
// CHECK-LABEL: @fold_reshape_load
//   CHECK-DAG:   %[[SRC_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(0){{.*}} memref<3x3x1x96xf32
//   CHECK-DAG:   %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC_SUBSPAN]]{{.*}} into memref<864xf32
//   CHECK-DAG:   %[[EXPAND:.+]] = memref.expand_shape %[[COLLAPSE]]{{.*}} into memref<3x3x96xf32
//       CHECK:   %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[EXPAND]]
//  CHECK-SAME:     memref<3x3x96xf32, #hal.descriptor_type<storage_buffer>> -> tensor<3x3x96xf32>
//       CHECK:   %[[DEST_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(1)
//       CHECK:   %[[BARRIER:.+]] = util.optimization_barrier %[[LOAD]]
//       CHECK:   iree_codegen.store_to_buffer %[[BARRIER]], %[[DEST_SUBSPAN]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fold_reshape_store() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<3x3x1x96xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<3x3x96xf32, #hal.descriptor_type<storage_buffer>>
  %2 = iree_codegen.load_from_buffer %0 : memref<3x3x1x96xf32, #hal.descriptor_type<storage_buffer>> -> tensor<3x3x1x96xf32>
  %barrier = util.optimization_barrier %2 : tensor<3x3x1x96xf32>
  %collapsed = tensor.collapse_shape %barrier [[0, 1, 2, 3]] : tensor<3x3x1x96xf32> into tensor<864xf32>
  %expanded = tensor.expand_shape %collapsed [[0, 1, 2]] output_shape [3, 3, 96] : tensor<864xf32> into tensor<3x3x96xf32>
  iree_codegen.store_to_buffer %expanded, %1 : tensor<3x3x96xf32> into memref<3x3x96xf32, #hal.descriptor_type<storage_buffer>>
  return
}
// CHECK-LABEL: @fold_reshape_store
//   CHECK-DAG:   %[[SRC_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(0)
//   CHECK-DAG:   %[[DEST_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(1){{.*}} memref<3x3x96xf32
//   CHECK-DAG:   %[[COLLAPSE:.+]] = memref.collapse_shape %[[DEST_SUBSPAN]]{{.*}} into memref<864xf32
//   CHECK-DAG:   %[[EXPAND:.+]] = memref.expand_shape %[[COLLAPSE]]{{.*}} into memref<3x3x1x96xf32
//       CHECK:   %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[SRC_SUBSPAN]]
//       CHECK:   %[[BARRIER:.+]] = util.optimization_barrier %[[LOAD]]
//       CHECK:   iree_codegen.store_to_buffer %[[BARRIER]], %[[EXPAND]]
//  CHECK-SAME:     tensor<3x3x1x96xf32> into memref<3x3x1x96xf32, #hal.descriptor_type<storage_buffer>>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fold_reshape_with_slice_load() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<6x3x1x96xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<3x3x96xf32, #hal.descriptor_type<storage_buffer>>
  %subview = memref.subview %0[3, 0, 0, 0] [3, 3, 1, 96] [1, 1, 1, 1] : memref<6x3x1x96xf32, #hal.descriptor_type<storage_buffer>> to memref<3x3x1x96xf32, strided<[288, 96, 96, 1], offset: 864>, #hal.descriptor_type<storage_buffer>>
  %2 = iree_codegen.load_from_buffer %subview : memref<3x3x1x96xf32, strided<[288, 96, 96, 1], offset: 864>, #hal.descriptor_type<storage_buffer>> -> tensor<3x3x1x96xf32>
  %collapsed = tensor.collapse_shape %2 [[0, 1, 2, 3]] : tensor<3x3x1x96xf32> into tensor<864xf32>
  %expanded = tensor.expand_shape %collapsed [[0, 1, 2]] output_shape [3, 3, 96] : tensor<864xf32> into tensor<3x3x96xf32>
  %barrier = util.optimization_barrier %expanded : tensor<3x3x96xf32>
  iree_codegen.store_to_buffer %barrier, %1 : tensor<3x3x96xf32> into memref<3x3x96xf32, #hal.descriptor_type<storage_buffer>>
  return
}
// CHECK-LABEL: @fold_reshape_with_slice_load
//   CHECK-DAG:   %[[SRC_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(0)
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[SRC_SUBSPAN]]
//  CHECK-SAME:     memref<3x3x1x96xf32, strided<[288, 96, 96, 1], offset: 864>
//       CHECK:   %[[COLLAPSE:.+]] = memref.collapse_shape %[[SUBVIEW]]
//  CHECK-SAME:     into memref<864xf32, strided<[1], offset: 864>
//       CHECK:   %[[EXPAND:.+]] = memref.expand_shape %[[COLLAPSE]]
//  CHECK-SAME:     into memref<3x3x96xf32, strided<[288, 96, 1], offset: 864>
//       CHECK:   iree_codegen.load_from_buffer %[[EXPAND]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fold_reshape_with_slice_store() {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<3x3x1x96xf32, #hal.descriptor_type<storage_buffer>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<6x3x96xf32, #hal.descriptor_type<storage_buffer>>
  %subview = memref.subview %1[3, 0, 0] [3, 3, 96] [1, 1, 1] : memref<6x3x96xf32, #hal.descriptor_type<storage_buffer>> to memref<3x3x96xf32, strided<[288, 96, 1], offset: 864>, #hal.descriptor_type<storage_buffer>>
  %2 = iree_codegen.load_from_buffer %0 : memref<3x3x1x96xf32, #hal.descriptor_type<storage_buffer>> -> tensor<3x3x1x96xf32>
  %barrier = util.optimization_barrier %2 : tensor<3x3x1x96xf32>
  %collapsed = tensor.collapse_shape %barrier [[0, 1, 2, 3]] : tensor<3x3x1x96xf32> into tensor<864xf32>
  %expanded = tensor.expand_shape %collapsed [[0, 1, 2]] output_shape [3, 3, 96] : tensor<864xf32> into tensor<3x3x96xf32>
  iree_codegen.store_to_buffer %expanded, %subview : tensor<3x3x96xf32> into memref<3x3x96xf32, strided<[288, 96, 1], offset: 864>, #hal.descriptor_type<storage_buffer>>
  return
}
// CHECK-LABEL: @fold_reshape_with_slice_store
//   CHECK-DAG:   %[[DEST_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(1)
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[DEST_SUBSPAN]]
//  CHECK-SAME:     memref<3x3x96xf32, strided<[288, 96, 1], offset: 864>
//       CHECK:   %[[COLLAPSE:.+]] = memref.collapse_shape %[[SUBVIEW]]
//  CHECK-SAME:     into memref<864xf32, strided<[1], offset: 864>
//       CHECK:   %[[EXPAND:.+]] = memref.expand_shape %[[COLLAPSE]]
//  CHECK-SAME:     into memref<3x3x1x96xf32, strided<[288, 96, 96, 1], offset: 864>
//       CHECK:   iree_codegen.store_to_buffer {{.*}}, %[[EXPAND]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fold_dynamic_reshape_load() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%0, %1}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%2, %3}
  %6 = iree_codegen.load_from_buffer %4 : memref<?x?xf32, #hal.descriptor_type<storage_buffer>> -> tensor<?x?xf32>
  %collapsed = tensor.collapse_shape %6 [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
  %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [%2, %3] : tensor<?xf32> into tensor<?x?xf32>
  %barrier = util.optimization_barrier %expanded : tensor<?x?xf32>
  iree_codegen.store_to_buffer %barrier, %5 : tensor<?x?xf32> into memref<?x?xf32, #hal.descriptor_type<storage_buffer>>
  return
}
// CHECK-LABEL: @fold_dynamic_reshape_load
//   CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load{{.*}} ordinal(2) : index
//   CHECK-DAG:   %[[D1:.+]] = hal.interface.constant.load{{.*}} ordinal(3) : index
//   CHECK-DAG:   %[[SRC_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(0)
//   CHECK-DAG:   %[[COLLAPSE:.+]] = memref.collapse_shape %[[SRC_SUBSPAN]]
//   CHECK-DAG:   %[[EXPAND:.+]] = memref.expand_shape %[[COLLAPSE]]
//  CHECK-SAME:     output_shape [%[[D0]], %[[D1]]]
//   CHECK-DAG:   %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[EXPAND]]
//   CHECK-DAG:   %[[DEST_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(1)
//   CHECK-DAG:   %[[BARRIER:.+]] = util.optimization_barrier %[[LOAD]]
//       CHECK:   iree_codegen.store_to_buffer %[[BARRIER]], %[[DEST_SUBSPAN]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fold_dynamic_reshape_store() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%0, %1}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<?x?xf32, #hal.descriptor_type<storage_buffer>>{%2, %3}
  %6 = iree_codegen.load_from_buffer %4 : memref<?x?xf32, #hal.descriptor_type<storage_buffer>> -> tensor<?x?xf32>
  %barrier = util.optimization_barrier %6 : tensor<?x?xf32>
  %collapsed = tensor.collapse_shape %barrier [[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
  %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [%2, %3] : tensor<?xf32> into tensor<?x?xf32>
  iree_codegen.store_to_buffer %expanded, %5 : tensor<?x?xf32> into memref<?x?xf32, #hal.descriptor_type<storage_buffer>>
  return
}
// CHECK-LABEL: @fold_dynamic_reshape_store
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//   CHECK-DAG:   %[[SRC_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(0)
//   CHECK-DAG:   %[[LOAD:.+]] = iree_codegen.load_from_buffer %[[SRC_SUBSPAN]]
//   CHECK-DAG:   %[[BARRIER:.+]] = util.optimization_barrier %[[LOAD]]
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[BARRIER]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[BARRIER]], %[[C1]]
//   CHECK-DAG:   %[[DEST_SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(1)
//   CHECK-DAG:   %[[COLLAPSE:.+]] = memref.collapse_shape %[[DEST_SUBSPAN]]
//   CHECK-DAG:   %[[EXPAND:.+]] = memref.expand_shape %[[COLLAPSE]]
//  CHECK-SAME:     output_shape [%[[D0]], %[[D1]]]
//       CHECK:   iree_codegen.store_to_buffer %[[BARRIER]], %[[EXPAND]]
