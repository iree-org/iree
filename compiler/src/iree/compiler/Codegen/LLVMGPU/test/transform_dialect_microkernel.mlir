// RUN: iree-opt %s --split-input-file -iree-transform-dialect-interpreter -transform-dialect-drop-schedule | FileCheck %s

!tlhs = tensor<2048x2048xf32>
!rlhs = tensor<2048x2048xf32>
!tres = tensor<2048x2048xf32>

func.func @fill_matmul(%lhs : !tlhs,   %rhs : !rlhs,   %accum : !tres) -> !tres {
  %cst = arith.constant 40.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%accum : !tres) -> !tres  
  %matmul = linalg.matmul 
                ins(%lhs, %rhs : !tlhs, !rlhs) 
                outs(%out : !tres) -> !tres
  return %matmul: !tres
}

transform.sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):
    // Fuse fill + matmul
    // ==========================================
    %fuseOp = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation

    %matmulOp = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    %forall, %tiled_matmul = transform.structured.tile_to_forall_op %matmulOp tile_sizes [128, 128] 
      ( mapping = [#gpu.block<x>, #gpu.block<y>] )

    transform.structured.fuse_into_containing_op %fuseOp into %forall

    // Transform tiled matmul to microkernel
    // ==========================================
    %tiledMatmulOp = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation

    transform.iree.microkernel_gpu_matmul %tiledMatmulOp, stages = 3, tile_k = 32 : (!pdl.operation) -> !pdl.operation
}

// CHECK-LABEL: @fill_matmul
//       CHECK:   bufferization.alloc_tensor() : tensor<128x128xf32>
//       CHECK:   bufferization.alloc_tensor() : tensor<8192xf32>
//       CHECK:   iree_codegen.ukernel.generic "__iree_matmul_tf32_tf32_f32_128_128_32_3_true_true" 


// -----

!tlhs = tensor<2048x2048xf32>
!rlhs = tensor<2048x2048xf32>
!tres = tensor<2048x2048xf32>

func.func @fill_matmul(%lhs : !tlhs,   %rhs : !rlhs,   %accum : !tres) -> !tres {
  %cst = arith.constant 40.0 : f32
  %out = linalg.fill ins(%cst : f32) outs(%accum : !tres) -> !tres  
  %matmul = linalg.matmul 
                ins(%lhs, %rhs : !tlhs, !rlhs) 
                outs(%out : !tres) -> !tres
  return %matmul: !tres
}

transform.sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):
    // Fuse fill + matmul
    // ==========================================
    %fuseOp = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!pdl.operation) -> !pdl.operation

    %matmulOp = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    %forall, %tiled_matmul = transform.structured.tile_to_forall_op %matmulOp tile_sizes [128, 128] 
      ( mapping = [#gpu.block<x>, #gpu.block<y>] )

    transform.structured.fuse_into_containing_op %fuseOp into %forall

    // Transform tiled matmul to microkernel
    // ==========================================
    %tiledMatmulOp = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation

    transform.iree.microkernel_gpu_matmul %tiledMatmulOp, name = "very_fast_code", stages = 3, tile_k = 32 : (!pdl.operation) -> !pdl.operation
}

// CHECK-LABEL: @fill_matmul
//       CHECK:   bufferization.alloc_tensor() : tensor<128x128xf32>
//       CHECK:   bufferization.alloc_tensor() : tensor<8192xf32>
//       CHECK:   iree_codegen.ukernel.generic "very_fast_code" 

