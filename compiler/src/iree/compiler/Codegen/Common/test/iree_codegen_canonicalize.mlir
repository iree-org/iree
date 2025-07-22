// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-canonicalize))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @fold_dynamic_trivial_subview(%size: index) -> memref<?xf32, #hal.descriptor_type<storage_buffer>> {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<?xf32, #hal.descriptor_type<storage_buffer>>{%size}
  %assume_align = memref.assume_alignment %0, 64 : memref<?xf32, #hal.descriptor_type<storage_buffer>>
  %subview = memref.subview %assume_align[0] [%size] [1] : memref<?xf32, #hal.descriptor_type<storage_buffer>> to memref<?xf32, #hal.descriptor_type<storage_buffer>>
  return %subview : memref<?xf32, #hal.descriptor_type<storage_buffer>>
}
// CHECK-LABEL: @fold_dynamic_trivial_subview
//  CHECK-SAME:   %[[SIZE:.+]]: index
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan{{.*}} binding(0)
//       CHECK:   %[[ASSUME_ALIGN:.+]] = memref.assume_alignment %[[SUBSPAN]]
//   CHECK-NOT:   memref.subview
//       CHECK:   return %[[ASSUME_ALIGN]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @no_fold_dynamic_real_subview(%size: index, %slice_size: index) -> memref<?xf32, #hal.descriptor_type<storage_buffer>> {
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<?xf32, #hal.descriptor_type<storage_buffer>>{%size}
  %assume_align = memref.assume_alignment %0, 64 : memref<?xf32, #hal.descriptor_type<storage_buffer>>
  %subview = memref.subview %assume_align[0] [%slice_size] [1] : memref<?xf32, #hal.descriptor_type<storage_buffer>> to memref<?xf32, #hal.descriptor_type<storage_buffer>>
  return %subview : memref<?xf32, #hal.descriptor_type<storage_buffer>>
}
// CHECK-LABEL: @no_fold_dynamic_real_subview
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview
//       CHECK:   return %[[SUBVIEW]]
