// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule | FileCheck %s

builtin.module attributes { transform.with_named_sequence } {
hal.executable private @pad_matmul_static_dispatch_0  {
  builtin.module {
    func.func @pad_matmul_static_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<250x500xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<500x1020xf32>>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<250x1020xf32>>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<250x500xf32>> -> tensor<250x500xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [500, 1020], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<500x1020xf32>> -> tensor<500x1020xf32>

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

      flow.dispatch.tensor.store %6, %2, offsets=[0, 0], sizes=[250, 1020], strides=[1, 1] : tensor<250x1020xf32> -> !flow.dispatch.tensor<readwrite:tensor<250x1020xf32>>
      return 
    }
  }
}

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.consumed}) {
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op: (!transform.any_op) -> !transform.any_op
    %func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.yield 
  }
} // module
