// RUN: iree-dialects-opt %s -linalg-transform-interp -canonicalize | FileCheck %s

// CHECK-LABEL: func @parallel_insert_slice_no_conflict(
//  CHECK-SAME:     %[[idx:.*]]: index, %[[idx2:.*]]: index,
//  CHECK-SAME:     %[[arg1:.*]]: memref<?xf32, #{{.*}}>,
//  CHECK-SAME:     %[[arg2:.*]]: memref<?xf32, #{{.*}}>
func @parallel_insert_slice_no_conflict(
    %idx: index, %idx2: index,
    %arg1: tensor<?xf32> {bufferization.writable=true},
    %arg2: tensor<?xf32> {bufferization.writable=true}) -> (tensor<?xf32>, f32)
{
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: iree_linalg_ext.in_parallel %[[idx2]]  -> ()
  %2 = iree_linalg_ext.in_parallel %idx2  -> (tensor<?xf32>) {
    ^bb0(%arg3: index):  // no predecessors
      // CHECK: %[[subview:.*]] = memref.subview %[[arg2]][5] [%[[idx]]] [1]
      %6 = tensor.extract_slice %arg2[5] [%idx] [%c1] : tensor<?xf32> to tensor<?xf32>
      // CHECK: linalg.fill ins(%{{.*}}) outs(%[[subview]] : memref<?xf32
      %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<?xf32>) -> tensor<?xf32>

      // CHECK: iree_linalg_ext.perform_concurrently
      // CHECK-NOT: parallel_insert_slice
      iree_linalg_ext.perform_concurrently {
        iree_linalg_ext.parallel_insert_slice %8 into %arg2[5] [%idx] [%c1] : tensor<?xf32> into tensor<?xf32>
      }
  }

  // CHECK: %[[load:.*]] = memref.load %[[arg2]]
  %f = tensor.extract %2[%c0] : tensor<?xf32>

  // CHECK: return %[[load]] : f32
  return %2, %f : tensor<?xf32>, f32
}

// CHECK-LABEL: func @parallel_insert_slice_with_conflict(
//  CHECK-SAME:     %[[idx:.*]]: index, %[[idx2:.*]]: index,
//  CHECK-SAME:     %[[arg1:.*]]: memref<?xf32, #{{.*}}>,
//  CHECK-SAME:     %[[arg2:.*]]: memref<?xf32, #{{.*}}>
func @parallel_insert_slice_with_conflict(
    %idx: index, %idx2: index,
    %arg1: tensor<?xf32> {bufferization.writable=true},
    %arg2: tensor<?xf32> {bufferization.writable=true}) -> (f32, f32)
{
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // The parallel_insert_slice_op bufferizes out-of-place, so we need an allocation.
  // CHECK: %[[alloc1:.*]] = memref.alloc
  // CHECK: linalg.generic {{.*}} ins(%[[arg2]]{{.*}}outs(%[[alloc1]]

  // CHECK: iree_linalg_ext.in_parallel %[[idx2]]  -> ()
  %2 = iree_linalg_ext.in_parallel %idx2  -> (tensor<?xf32>) {
    ^bb0(%arg3: index):  // no predecessors
      // Another alloc for the extract_slice op.
      // CHECK: %[[alloc2:.*]] = memref.alloc
      %6 = tensor.extract_slice %arg2[5] [%idx] [%c1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: linalg.fill ins(%{{.*}}) outs(%[[alloc2]] : memref<?xf32
      %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<?xf32>) -> tensor<?xf32>

      // Now the copy of the actual insert_slice.
      // CHECK: %[[subview1:.*]] = memref.subview %[[alloc1]][5] [%[[idx]]] [1]
      //
      // CHECK: linalg.generic {{.*}} ins(%[[alloc2]]{{.*}}outs(%[[subview1]]
      // CHECK: memref.dealloc %[[alloc2]]

      // The terminator is empty.
      // CHECK: iree_linalg_ext.perform_concurrently
      // CHECK-NOT: parallel_insert_slice
      iree_linalg_ext.perform_concurrently {
        iree_linalg_ext.parallel_insert_slice %8 into %arg2[5] [%idx] [%c1] : tensor<?xf32> into tensor<?xf32>
      }
  }

  // CHECK: %[[load:.*]] = memref.load %[[arg2]]
  // CHECK: %[[load2:.*]] = memref.load %[[alloc1]]
  // CHECK: memref.dealloc %[[alloc1]]
  %f = tensor.extract %arg2[%c0] : tensor<?xf32>
  %f2 = tensor.extract %2[%c0] : tensor<?xf32>

  // CHECK: return %[[load2]], %[[load]] : f32, f32
  return %f2, %f : f32, f32
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_target_2 : benefit(1) {
    %0 = operation "func"
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    bufferize
  }
}
