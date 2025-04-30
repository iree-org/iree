// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @pad_matmul_static_dispatch_0  {
  builtin.module {
    func.func @pad_matmul_static_dispatch_0(%arg0: tensor<250x500xf32>, %arg1: tensor<500x1020xf32>) -> tensor<250x1020xf32> {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<250x500xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<500x1020xf32>>
      %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<250x500xf32>> -> tensor<250x500xf32>
      %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [500, 1020], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<500x1020xf32>> -> tensor<500x1020xf32>

      %50 = tensor.empty() : tensor<250x1020xf32>
      %cst = arith.constant 0.000000e+00 : f32
      %5 = linalg.fill ins(%cst : f32) outs(%50 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
      // CHECK:      %[[CST:.+]] = arith.constant 0.000000e+00 : f32
      // CHECK:      %[[D0:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) alignment(64)
      // CHECK:      %[[D1:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) alignment(64)
      // CHECK:      %[[D2:.+]] = iree_tensor_ext.dispatch.tensor.load %[[D0]], offsets = [0, 0], sizes = [250, 500]
      // CHECK:      %[[D3:.+]] = iree_tensor_ext.dispatch.tensor.load %[[D1]], offsets = [0, 0], sizes = [500, 1020]
      // CHECK:      %[[D4:.+]] = tensor.empty() : tensor<250x1020xf32>
      // CHECK-NEXT: %[[D5:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[D4]] : tensor<250x1020xf32>) -> tensor<250x1020xf32>
      // CHECK-NEXT: %[[D6:.+]] = bufferization.alloc_tensor() copy(%[[D2]]) : tensor<250x500xf32>
      // CHECK-NEXT: %[[D7:.+]] = bufferization.alloc_tensor() copy(%[[D3]]) : tensor<500x1020xf32>
      // CHECK-NEXT: %[[D8:.+]] = linalg.matmul ins(%[[D6]], %[[D7]] : tensor<250x500xf32>, tensor<500x1020xf32>)
      // CHECK-SAME:                 outs(%[[D5]] : tensor<250x1020xf32>) -> tensor<250x1020xf32>
      %6 = linalg.matmul ins(%3, %4 : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%5 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
      return %6: tensor<250x1020xf32>
    }
  }

  builtin.module attributes { transform.with_named_sequence } {
    transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
      %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
        : (!transform.any_op) -> !transform.any_op
      %promoted_matmul, %alloc_0, %alloc_1 =
        transform.iree.promote_operands %matmul [0, 1]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

      // Late canonicalizations to cleanup and pass the checks.
      %func_op = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %func_op {
        transform.apply_patterns.iree.fold_fill_into_pad
        transform.apply_patterns.linalg.tiling_canonicalization
        transform.apply_patterns.scf.for_loop_canonicalization
        transform.apply_patterns.canonicalization
      } : !transform.any_op
      transform.iree.apply_licm %func_op : !transform.any_op
      transform.apply_cse to %func_op : !transform.any_op
      transform.yield
    } // @__transform_main
  } // module
}
