// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule | FileCheck %s

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

  %50 = tensor.empty() : tensor<250x1020xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %5 = linalg.fill ins(%cst : f32) outs(%50 : tensor<250x1020xf32>) -> tensor<250x1020xf32>

  // CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<250x1020xf32, #hal.descriptor_type<storage_buffer>>)
  // CHECK: memref.alloc() {alignment = 64 : i64} : memref<250x500xf32, #gpu.address_space<workgroup>>
  // CHECK: gpu.barrier
  // CHECK: linalg.generic
  // CHECK: gpu.barrier
  // CHECK-NEXT: linalg.matmul{{.*}}ins(%{{.*}} : memref<250x500xf32, #gpu.address_space<workgroup>>, memref<500x1020xf32, #hal.descriptor_type<storage_buffer>>) outs(%{{.*}} : memref<250x1020xf32, #hal.descriptor_type<storage_buffer>>)
  %p = bufferization.alloc_tensor() copy(%3) : tensor<250x500xf32>
  %6 = linalg.matmul ins(%p, %4 : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%5 : tensor<250x1020xf32>) -> tensor<250x1020xf32>

  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets=[0, 0], sizes=[250, 1020], strides=[1, 1] : tensor<250x1020xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<250x1020xf32>>
  return
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.iree.eliminate_empty_tensors %func : (!transform.any_op) -> ()
    %_ = transform.iree.bufferize { target_gpu } %func: (!transform.any_op) -> !transform.any_op
    transform.yield
  } // @__transform_main
} // module
