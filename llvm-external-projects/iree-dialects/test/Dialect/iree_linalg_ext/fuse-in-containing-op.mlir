// RUN: iree-dialects-opt %s  --transform-dialect-interpreter --split-input-file | FileCheck %s

#map0 = affine_map<()[s0] -> (64 ceildiv s0)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0] -> (-(d0 * s0) + 64, s0)>

module {
  // CHECK-LABEL: func.func @fuse_static
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<64xf32>
  //  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: tensor<64xf32>
  func.func @fuse_static(%arg0: index, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> {
    %cst = arith.constant 4.200000e+01 : f32
    %0 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<64xf32>) -> tensor<64xf32>
    %1 = affine.apply #map0()[%arg0]
    // CHECK: scf.foreach_thread
    %2 = scf.foreach_thread (%arg3) in (%1)  -> (tensor<64xf32>) {
      // CHECK:    %[[OFFSET:.*]] = affine.apply
      // CHECK:    %[[SIZE:.*]] = affine.min
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.min #map2(%arg3)[%arg0]
      %5 = tensor.extract_slice %arg2[%3] [%4] [1] : tensor<64xf32> to tensor<?xf32>

      // CHECK:    %[[T0:.*]] = tensor.extract_slice %[[IN]][%[[OFFSET]]] [%[[SIZE]]] [{{.*}}]
      // CHECK:    %[[T1:.*]] = linalg.fill {{.*}} outs(%[[T0]]
      %6 = tensor.extract_slice %0[%3] [%4] [1] : tensor<64xf32> to tensor<?xf32>

      // CHECK:    %[[T2:.*]] = linalg.elemwise_unary ins(%[[T1]]
      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.foreach_thread.perform_concurrently {
        scf.foreach_thread.parallel_insert_slice %7 into %arg2[%3] [%4] [1] : tensor<?xf32> into tensor<64xf32>
      }
    }
    func.return %2 : tensor<64xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    pdl.pattern @match_fill : benefit(1) {
      %0 = operands
      %1 = types
      %2 = operation "linalg.fill"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
      rewrite %2 with "transform.dialect"
    }
    pdl.pattern @match_foreach_thread : benefit(1) {
      %0 = operands
      %1 = types
      %2 = operation "scf.foreach_thread"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
      rewrite %2 with "transform.dialect"
    }
    transform.structured.canonicalized_sequence %arg0 {
    ^bb1(%arg1: !pdl.operation):
      %0 = pdl_match @match_fill in %arg1
      %1 = pdl_match @match_foreach_thread in %arg1
      fuse_into_containing_op %0 into %1
    }
  }
}

// -----

#map0 = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0)[s0, s1] -> (-(d0 * s1) + s0, s1)>

module {
  // CHECK-LABEL: func.func @fuse_dynamic
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?xf32>
  //  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: tensor<?xf32>
  func.func @fuse_dynamic(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %cst = arith.constant 4.200000e+01 : f32
    %c0 = arith.constant 0 : index
    %0 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<?xf32>) -> tensor<?xf32>
    // TODO: Choosing %arg2 here complicates the size computation.
    %d0 = tensor.dim %arg1, %c0 : tensor<?xf32>
    %1 = affine.apply #map0()[%d0, %arg0]
    // CHECK: scf.foreach_thread
    %2 = scf.foreach_thread (%arg3) in (%1)  -> (tensor<?xf32>) {
      // CHECK:    %[[OFFSET:.*]] = affine.apply
      // CHECK:    %[[SIZE:.*]] = affine.min
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.min #map2(%arg3)[%d0, %arg0]
      %5 = tensor.extract_slice %arg2[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK:    %[[T0:.*]] = tensor.extract_slice %[[IN]][%[[OFFSET]]] [%[[SIZE]]] [{{.*}}]
      // CHECK:    %[[T1:.*]] = linalg.fill {{.*}} outs(%[[T0]]
      %6 = tensor.extract_slice %0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK:    %[[T2:.*]] = linalg.elemwise_unary ins(%[[T1]]
      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.foreach_thread.perform_concurrently {
        scf.foreach_thread.parallel_insert_slice %7 into %arg2[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    func.return %2 : tensor<?xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    pdl.pattern @match_fill : benefit(1) {
      %0 = operands
      %1 = types
      %2 = operation "linalg.fill"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
      rewrite %2 with "transform.dialect"
    }
    pdl.pattern @match_foreach_thread : benefit(1) {
      %0 = operands
      %1 = types
      %2 = operation "scf.foreach_thread"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
      rewrite %2 with "transform.dialect"
    }
    transform.structured.canonicalized_sequence %arg0 {
    ^bb1(%arg1: !pdl.operation):
      %0 = pdl_match @match_fill in %arg1
      %1 = pdl_match @match_foreach_thread in %arg1
      fuse_into_containing_op %0 into %1
    }
  }
}
