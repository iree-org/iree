// RUN: iree-opt %s --iree-transform-dialect-interpreter --transform-dialect-drop-schedule | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @pad_matmul_static_dispatch_0() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<250x500xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<500x1020xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<250x1020xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<250x500xf32>> -> tensor<250x500xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [500, 1020], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<500x1020xf32>> -> tensor<500x1020xf32>
  %5 = tensor.empty() : tensor<250x1020xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<250x1020xf32>) -> tensor<250x1020xf32>

  //      CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<250x1020xf32, #hal.descriptor_type<storage_buffer>>)
  // CHECK-NEXT: linalg.matmul
  // CHECK-SAME:   ins(%{{.*}} : memref<250x500xf32, #hal.descriptor_type<storage_buffer>>, memref<500x1020xf32, #hal.descriptor_type<storage_buffer>>)
  // CHECK-SAME:   outs(%{{.*}} : memref<250x1020xf32, #hal.descriptor_type<storage_buffer>>)
  // CHECK-NEXT: return

  %7 = linalg.matmul ins(%3, %4 : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%6 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [250, 1020], strides = [1, 1] : tensor<250x1020xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<250x1020xf32>>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %8 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.print %8 : !transform.any_op
    transform.iree.eliminate_empty_tensors %8 : (!transform.any_op) -> ()
    %9 = transform.iree.bufferize %8 : (!transform.any_op) -> !transform.any_op
    %10 = transform.structured.match ops{["func.func"]} in %9 : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %10 {
      transform.apply_patterns.canonicalization
    } : !transform.op<"func.func">
    transform.yield
  }
}
