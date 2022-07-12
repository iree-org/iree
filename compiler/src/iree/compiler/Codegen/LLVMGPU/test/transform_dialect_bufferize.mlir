// RUN: iree-opt %s -transform-dialect-interpreter -transform-dialect-drop-schedule | FileCheck %s

func.func @pad_matmul_static_dispatch_0(%arg0: tensor<250x500xf32>, %arg1: tensor<500x1020xf32>) -> tensor<250x1020xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:250x500xf32>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:500x1020xf32>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:250x1020xf32>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [250, 500], strides = [1, 1] : !flow.dispatch.tensor<readonly:250x500xf32> -> tensor<250x500xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [500, 1020], strides = [1, 1] : !flow.dispatch.tensor<readonly:500x1020xf32> -> tensor<500x1020xf32>

  %50 = linalg.init_tensor [250, 1020] : tensor<250x1020xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %5 = linalg.fill ins(%cst : f32) outs(%50 : tensor<250x1020xf32>) -> tensor<250x1020xf32>

  //      CHECK: memref.alloc() {alignment = 128 : i64} : memref<250x1020xf32, 3>
  // CHECK-NEXT: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : memref<250x1020xf32, 3>)
  // CHECK-NEXT: linalg.matmul{{.*}}ins(%{{.*}} : memref<250x500xf32>, memref<500x1020xf32>) outs(%{{.*}} : memref<250x1020xf32, 3>)
  // CHECK-NEXT: bufferization.to_tensor %{{.*}} : memref<250x1020xf32, 3>
  // CHECK-NEXT: memref.dealloc %{{.*}} : memref<250x1020xf32, 3>

  %6 = linalg.matmul ins(%3, %4 : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%5 : tensor<250x1020xf32>) -> tensor<250x1020xf32>

  // Returning a tensor force the allocation that we want to test here: and alloca with memory space "3" for GPU shared memory.
  return %6: tensor<250x1020xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_matmul_target : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_matmul_target in %arg1
    transform.iree.bufferize { target_gpu }
  }
}
