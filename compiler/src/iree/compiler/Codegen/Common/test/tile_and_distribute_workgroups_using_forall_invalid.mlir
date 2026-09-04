// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op, cse))" --split-input-file --verify-diagnostics %s

// Verify that a compute consumer left outside the distribution loop causes the
// pass to fail. The pack cannot be fused because the tiling is not perfect, but
// only the pack itself is allowed to remain outside the loop.

#config = #iree_cpu.lowering_config<distribution = [1, 16]>
func.func @unfused_compute_consumer(%arg0 : tensor<30xf32>) -> tensor<5x6xf32> {
  %empty = tensor.empty() : tensor<30xf32>
  %producer = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<30xf32>) outs(%empty : tensor<30xf32>)
      attrs = {lowering_config = #config} {
    ^bb0(%in : f32, %out : f32):
      %doubled = arith.addf %in, %in : f32
      linalg.yield %doubled : f32
  } -> tensor<30xf32>
  %packed_empty = tensor.empty() : tensor<5x6xf32>
  %pack = linalg.pack %producer outer_dims_perm = [0]
      inner_dims_pos = [0] inner_tiles = [6] into %packed_empty
      : tensor<30xf32> -> tensor<5x6xf32>
  %consumer_empty = tensor.empty() : tensor<5x6xf32>
  // expected-error @below {{failed to fuse consumers}}
  %consumer = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%pack : tensor<5x6xf32>) outs(%consumer_empty : tensor<5x6xf32>) {
    ^bb0(%in : f32, %out : f32):
      %doubled = arith.addf %in, %in : f32
      linalg.yield %doubled : f32
  } -> tensor<5x6xf32>
  return %consumer : tensor<5x6xf32>
}

// -----

// Verify all compute ops after distribution, even when consumer fusion
// succeeds. The configured op is distributed, while the disconnected compute
// op must not remain outside the distribution loop.

#config = #iree_cpu.lowering_config<distribution = [16]>
func.func @disconnected_compute_ops(
    %arg0 : tensor<32xf32>, %arg1 : tensor<32xf32>)
    -> (tensor<32xf32>, tensor<32xf32>) {
  %unfused_empty = tensor.empty() : tensor<32xf32>
  // expected-error @below {{failed to fuse consumers}}
  %unfused = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<32xf32>) outs(%unfused_empty : tensor<32xf32>) {
    ^bb0(%in : f32, %out : f32):
      %doubled = arith.addf %in, %in : f32
      linalg.yield %doubled : f32
  } -> tensor<32xf32>
  %distributed_empty = tensor.empty() : tensor<32xf32>
  %distributed = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg1 : tensor<32xf32>) outs(%distributed_empty : tensor<32xf32>)
      attrs = {lowering_config = #config} {
    ^bb0(%in : f32, %out : f32):
      %doubled = arith.addf %in, %in : f32
      linalg.yield %doubled : f32
  } -> tensor<32xf32>
  return %unfused, %distributed : tensor<32xf32>, tensor<32xf32>
}

// -----

// A producer needed exclusively by the distribution loop may remain outside
// until a later tiling level fuses it. Distribution only tiles the second
// dimension, which is absent from the producer's iteration space. The
// preceding thread-mapped forall must not be mistaken for the distribution
// loop.

#producer_config = #iree_cpu.lowering_config<distribution = [0, 16]>
func.func @producer_consumed_by_distribution(
    %arg0 : tensor<8xf32>, %arg1 : tensor<8x32xf32>)
    -> tensor<8x32xf32> {
  %c7 = arith.constant 7 : index
  %thread_result = scf.forall (%thread) in (8)
      shared_outs(%thread_out = %arg0) -> tensor<8xf32> {
    %reverse = arith.subi %c7, %thread : index
    %slice = tensor.extract_slice %arg0[%reverse] [1] [1]
        : tensor<8xf32> to tensor<1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %thread_out[%thread] [1] [1]
          : tensor<1xf32> into tensor<8xf32>
    }
  } {mapping = [#gpu.thread<x>]}
  %producer_empty = tensor.empty() : tensor<8xf32>
  %producer = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%thread_result : tensor<8xf32>)
      outs(%producer_empty : tensor<8xf32>) {
    ^bb0(%in : f32, %out : f32):
      %doubled = arith.addf %in, %in : f32
      linalg.yield %doubled : f32
  } -> tensor<8xf32>
  %distributed_empty = tensor.empty() : tensor<8x32xf32>
  %distributed = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%producer, %arg1 : tensor<8xf32>, tensor<8x32xf32>)
      outs(%distributed_empty : tensor<8x32xf32>)
      attrs = {lowering_config = #producer_config} {
    ^bb0(%lhs : f32, %rhs : f32, %out : f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
  } -> tensor<8x32xf32>
  return %distributed : tensor<8x32xf32>
}

// -----

// Shape-only users can keep the original compute op alive after its tiled
// replacement is created. Canonicalize those users before verifying so the
// original op and its producer are not mistaken for unfused computation.

#dynamic_config = #iree_cpu.lowering_config<distribution = [16]>
func.func @shape_only_users_after_distribution(%arg0 : tensor<?xf32>)
    -> (tensor<?xf32>, index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
  %empty = tensor.empty(%dim) : tensor<?xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<?xf32>)
      -> tensor<?xf32>
  %root = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg0 : tensor<?xf32>) outs(%filled : tensor<?xf32>)
      attrs = {lowering_config = #dynamic_config} {
    ^bb0(%in : f32, %out : f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
  } -> tensor<?xf32>
  %result_dim = tensor.dim %root, %c0 : tensor<?xf32>
  return %root, %result_dim : tensor<?xf32>, index
}

// -----

// Verify compute ops inside a non-workgroup forall instead of skipping the
// whole loop.

#thread_test_config = #iree_cpu.lowering_config<distribution = [16]>
func.func @compute_in_thread_forall(
    %arg0 : tensor<32xf32>, %arg1 : tensor<32xf32>)
    -> (tensor<32xf32>, tensor<32xf32>) {
  %c16 = arith.constant 16 : index
  %thread_empty = tensor.empty() : tensor<32xf32>
  %thread_result = scf.forall (%thread) in (2)
      shared_outs(%thread_out = %thread_empty) -> tensor<32xf32> {
    %offset = arith.muli %thread, %c16 : index
    %slice = tensor.extract_slice %arg0[%offset] [16] [1]
        : tensor<32xf32> to tensor<16xf32>
    %compute_empty = tensor.empty() : tensor<16xf32>
    // expected-error @below {{failed to fuse consumers}}
    %compute = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%slice : tensor<16xf32>) outs(%compute_empty : tensor<16xf32>) {
      ^bb0(%in : f32, %out : f32):
        %doubled = arith.addf %in, %in : f32
        linalg.yield %doubled : f32
    } -> tensor<16xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %compute into %thread_out[%offset] [16] [1]
          : tensor<16xf32> into tensor<32xf32>
    }
  } {mapping = [#gpu.thread<x>]}
  %distributed_empty = tensor.empty() : tensor<32xf32>
  %distributed = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%arg1 : tensor<32xf32>) outs(%distributed_empty : tensor<32xf32>)
      attrs = {lowering_config = #thread_test_config} {
    ^bb0(%in : f32, %out : f32):
      %doubled = arith.addf %in, %in : f32
      linalg.yield %doubled : f32
  } -> tensor<32xf32>
  return %thread_result, %distributed : tensor<32xf32>, tensor<32xf32>
}
