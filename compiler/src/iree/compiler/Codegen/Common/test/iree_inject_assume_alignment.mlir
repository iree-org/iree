// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-inject-assume-alignment))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @bindings() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<32x32xf32>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) offset(%c0) : memref<32x32xf32>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(32) offset(%c0) : memref<32x32xf32>
  return
}
// CHECK-LABEL: @bindings()
// CHECK:         %[[BIND_0:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-NEXT:    %{{.+}} = memref.assume_alignment %[[BIND_0]], 64
// CHECK:         %[[BIND_1:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK-NEXT:    %{{.+}} = memref.assume_alignment %[[BIND_1]], 4
// CHECK:         %[[BIND_2:.+]] = hal.interface.binding.subspan {{.+}} binding(2)
// CHECK-NEXT:    %{{.+}} = memref.assume_alignment %[[BIND_2]], 32
