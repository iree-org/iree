// RUN: iree-dialects-opt %s  --transform-dialect-interpreter --split-input-file | FileCheck %s
// TODO(#11765): Fix and re-enable this.
// REQUIRES: dont-run

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
    %cst2 = arith.constant 4.300000e+01 : f32
    %0 = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
        outs(%arg1 : tensor<64xf32>) {
    ^bb0(%arg3: f32):
      linalg.yield %cst : f32
    } -> tensor<64xf32>
    %1 = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
        outs(%arg2 : tensor<64xf32>) {
    ^bb0(%arg3: f32):
      linalg.yield %cst : f32
    } -> tensor<64xf32>

    %2 = affine.apply #map0()[%arg0]
    // CHECK: scf.forall
    %3 = scf.forall (%arg3) in (%2) shared_outs(%O = %arg2) -> (tensor<64xf32>) {
      // CHECK:    %[[OFFSET:.*]] = affine.apply
      // CHECK:    %[[SIZE:.*]] = affine.min
      %4 = affine.apply #map1(%arg3)[%arg0]
      %5 = affine.min #map2(%arg3)[%arg0]
      %6 = tensor.extract_slice %0[%4] [%5] [1] : tensor<64xf32> to tensor<?xf32>

      // CHECK:    %[[T0:.*]] = tensor.extract_slice %[[IN]][%[[OFFSET]]] [%[[SIZE]]] [{{.*}}]
      // CHECK:    %[[T1:.*]] = linalg.generic {{.*}} outs(%[[T0]]
      // CHECK:    %[[T2:.*]] = tensor.extract_slice %[[OUT]][%[[OFFSET]]] [%[[SIZE]]] [{{.*}}]
      // CHECK:    %[[T3:.*]] = linalg.generic {{.*}} outs(%[[T2]]
      %7 = tensor.extract_slice %1[%4] [%5] [1] : tensor<64xf32> to tensor<?xf32>

      // CHECK:    %[[T4:.*]] = linalg.elemwise_unary ins(%[[T1]] {{.*}} outs(%[[T3]]
      %8 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%7 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %O[%4] [%5] [1] : tensor<?xf32> into tensor<64xf32>
      }
    }
    func.return %3 : tensor<64xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    pdl.pattern @match_elemwise : benefit(1) {
      %0 = operands
      %1 = types
      %2 = operation "linalg.elemwise_unary"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
      rewrite %2 with "transform.dialect"
    }
    pdl.pattern @match_in_parallel : benefit(1) {
      %0 = operands
      %1 = types
      %2 = operation "scf.forall"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
      rewrite %2 with "transform.dialect"
    }
    transform.sequence %arg0 failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
      %0 = pdl_match @match_elemwise in %arg1 : (!pdl.operation) -> !pdl.operation
      %1, %fusedOps:2 = fuse_producers %0 {operands_to_fuse=[0, 1]}
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
    %0 = linalg.generic
        {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
        outs(%arg1 : tensor<?xf32>) {
    ^bb0(%arg3: f32):
      linalg.yield %cst : f32
    } -> tensor<?xf32>
    // TODO: Choosing %arg2 here complicates the size computation.
    %d0 = tensor.dim %arg1, %c0 : tensor<?xf32>
    %1 = affine.apply #map0()[%d0, %arg0]
    // CHECK: scf.forall
    %2 = scf.forall (%arg3) in (%1) shared_outs(%O = %arg2) -> (tensor<?xf32>) {
      // CHECK:    %[[OFFSET:.*]] = affine.apply
      // CHECK:    %[[SIZE:.*]] = affine.min
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.min #map2(%arg3)[%d0, %arg0]
      %5 = tensor.extract_slice %arg2[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK:    %[[T0:.*]] = tensor.extract_slice %[[IN]][%[[OFFSET]]] [%[[SIZE]]] [{{.*}}]
      // CHECK:    %[[T1:.*]] = linalg.generic {{.*}} outs(%[[T0]]
      %6 = tensor.extract_slice %0[%3] [%4] [1] : tensor<?xf32> to tensor<?xf32>

      // CHECK:    %[[T2:.*]] = linalg.elemwise_unary ins(%[[T1]]
      %7 = linalg.elemwise_unary ins(%6 : tensor<?xf32>) outs(%5 : tensor<?xf32>) -> tensor<?xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %7 into %O[%3] [%4] [1] : tensor<?xf32> into tensor<?xf32>
      }
    }
    func.return %2 : tensor<?xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    pdl.pattern @match_elemwise : benefit(1) {
      %0 = operands
      %1 = types
      %2 = operation "linalg.elemwise_unary"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
      rewrite %2 with "transform.dialect"
    }
    pdl.pattern @match_in_parallel : benefit(1) {
      %0 = operands
      %1 = types
      %2 = operation "scf.forall"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
      rewrite %2 with "transform.dialect"
    }
    transform.sequence %arg0 failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
      %0 = pdl_match @match_elemwise in %arg1 : (!pdl.operation) -> !pdl.operation
      %1, %fusedOps = fuse_producers %0 {operands_to_fuse=[0]}
    }
  }
}
