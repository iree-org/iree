// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-and-fuse-producer-consumer{tiling-level=vector_reduction only-fuse-producer-input-operands=true}))" --split-input-file %s | FileCheck %s

// The row max is the same for every key tile, so it stays outside the
// reduction loop, while exp and sum are tiled inside it. Shapes are dynamic on
// purpose: with static shapes the slice of the max folds away before fusion.
#id = affine_map<(r, k) -> (r, k)>
#row = affine_map<(r, k) -> (r)>
#cfg = #iree_cpu.lowering_config<distribution = [32, 0], vector_common_parallel = [4, 0], vector_reduction = [0, 4]>
func.func @max_then_sum(%scores: tensor<?x?xf32>) -> tensor<?xf32> {
  %neg = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %rows = tensor.dim %scores, %c0 : tensor<?x?xf32>
  %empty = tensor.empty(%rows) : tensor<?xf32>
  %init_max = linalg.fill ins(%neg : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %max = linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]}
      ins(%scores : tensor<?x?xf32>) outs(%init_max : tensor<?xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %m = arith.maxnumf %x, %acc : f32
      linalg.yield %m : f32
  } -> tensor<?xf32>
  %init_sum = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %sum = linalg.generic {indexing_maps = [#id, #row, #row], iterator_types = ["parallel", "reduction"]}
      ins(%scores, %max : tensor<?x?xf32>, tensor<?xf32>) outs(%init_sum : tensor<?xf32>)
      attrs = {lowering_config = #cfg} {
    ^bb0(%x: f32, %m: f32, %acc: f32):
      %d = arith.subf %x, %m : f32
      %e = math.exp %d : f32
      %s = arith.addf %e, %acc : f32
      linalg.yield %s : f32
  } -> tensor<?xf32>
  return %sum : tensor<?xf32>
}

// CHECK-LABEL: func.func @max_then_sum
// CHECK-NOT: scf.for {{.*}} =
// CHECK: %[[MAX:.*]] = linalg.generic
// CHECK: arith.maxnumf
// CHECK: scf.for {{.*}} =
// CHECK-NOT: arith.maxnumf
// CHECK: %[[MAX_SLICE:.*]] = tensor.extract_slice %[[MAX]][
// CHECK: ins(%{{.*}}, %[[MAX_SLICE]] :
// CHECK: math.exp
// CHECK: arith.addf
// CHECK-NOT: arith.maxnumf
// CHECK: scf.yield
// CHECK-NEXT: }
// CHECK: return

// -----

// Each sum tile requests different rows of maxima, so producer fusion remains
// useful: compute only those rows rather than materializing all row maxima.
#id = affine_map<(r, k) -> (r, k)>
#row = affine_map<(r, k) -> (r)>
#vec = affine_map<(r) -> (r)>
#scalar = affine_map<(r) -> ()>
#cfg = #iree_cpu.lowering_config<distribution = [0], vector_reduction = [4]>
func.func @tile_dependent_max(%scores: tensor<8x16xf32>) -> tensor<f32> {
  %neg = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %init_max = linalg.fill ins(%neg : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %max = linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]}
      ins(%scores : tensor<8x16xf32>) outs(%init_max : tensor<8xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %m = arith.maxnumf %x, %acc : f32
      linalg.yield %m : f32
  } -> tensor<8xf32>
  %scalar_empty = tensor.empty() : tensor<f32>
  %init_sum = linalg.fill ins(%zero : f32) outs(%scalar_empty : tensor<f32>) -> tensor<f32>
  %sum = linalg.generic {indexing_maps = [#vec, #scalar], iterator_types = ["reduction"]}
      ins(%max : tensor<8xf32>) outs(%init_sum : tensor<f32>)
      attrs = {lowering_config = #cfg} {
    ^bb0(%x: f32, %acc: f32):
      %s = arith.addf %x, %acc : f32
      linalg.yield %s : f32
  } -> tensor<f32>
  return %sum : tensor<f32>
}
// CHECK-LABEL: func.func @tile_dependent_max
// CHECK-NOT: arith.maxnumf
// CHECK: scf.for
// CHECK: tensor.extract_slice %{{.*}}[%{{.*}}, 0] [4, 16]
// CHECK: arith.maxnumf
// CHECK: arith.addf
// CHECK: scf.yield
// CHECK-NEXT: }
// CHECK: return

// -----

// Every key tile uses the same scaled row bias. Unlike a reduction producer,
// this elementwise producer stays fused and is recomputed for each key tile.
#id = affine_map<(r, k) -> (r, k)>
#row = affine_map<(r, k) -> (r)>
#vec = affine_map<(r) -> (r)>
#cfg = #iree_cpu.lowering_config<distribution = [32, 0], vector_common_parallel = [4, 0], vector_reduction = [0, 4]>
func.func @invariant_elementwise_producer_stays_fused(%scores: tensor<?x?xf32>, %bias: tensor<?xf32>) -> tensor<?xf32> {
  %zero = arith.constant 0.0 : f32
  %scale = arith.constant 2.0 : f32
  %c0 = arith.constant 0 : index
  %rows = tensor.dim %scores, %c0 : tensor<?x?xf32>
  %empty = tensor.empty(%rows) : tensor<?xf32>
  %scaled_bias = linalg.generic {indexing_maps = [#vec, #vec], iterator_types = ["parallel"]}
      ins(%bias : tensor<?xf32>) outs(%empty : tensor<?xf32>) {
    ^bb0(%x: f32, %unused: f32):
      %scaled = arith.mulf %x, %scale : f32
      linalg.yield %scaled : f32
  } -> tensor<?xf32>
  %init_sum = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %sum = linalg.generic {indexing_maps = [#id, #row, #row], iterator_types = ["parallel", "reduction"]}
      ins(%scores, %scaled_bias : tensor<?x?xf32>, tensor<?xf32>) outs(%init_sum : tensor<?xf32>)
      attrs = {lowering_config = #cfg} {
    ^bb0(%x: f32, %bias_value: f32, %acc: f32):
      %d = arith.subf %x, %bias_value : f32
      %e = math.exp %d : f32
      %s = arith.addf %e, %acc : f32
      linalg.yield %s : f32
  } -> tensor<?xf32>
  return %sum : tensor<?xf32>
}

// CHECK-LABEL: func.func @invariant_elementwise_producer_stays_fused
// CHECK-NOT: arith.mulf
// CHECK: scf.for
// CHECK: %[[SCALED_BIAS:.*]] = linalg.generic
// CHECK: arith.mulf
// CHECK: ins(%{{.*}}, %[[SCALED_BIAS]] :
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: scf.yield
// CHECK-NEXT: }
// CHECK: return

// -----

// The channel max is the same for every (h, w) tile, so it stays outside both
// reduction loops.
#id = affine_map<(c, h, w) -> (c, h, w)>
#chan = affine_map<(c, h, w) -> (c)>
#max_cfg = #iree_cpu.lowering_config<vector_common_parallel = [4, 0, 0], vector_reduction = [0, 1, 4]>
#cfg = #iree_cpu.lowering_config<distribution = [8, 0, 0], vector_common_parallel = [4, 0, 0], vector_reduction = [0, 1, 4]>
func.func @max_then_sum_two_reduction_dims(%input: tensor<?x?x?xf32>) -> tensor<?xf32> {
  %neg = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %chans = tensor.dim %input, %c0 : tensor<?x?x?xf32>
  %empty = tensor.empty(%chans) : tensor<?xf32>
  %init_max = linalg.fill ins(%neg : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %max = linalg.generic {indexing_maps = [#id, #chan], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%input : tensor<?x?x?xf32>) outs(%init_max : tensor<?xf32>)
      attrs = {lowering_config = #max_cfg} {
    ^bb0(%in: f32, %acc: f32):
      %m = arith.maxnumf %in, %acc : f32
      linalg.yield %m : f32
  } -> tensor<?xf32>
  %init_sum = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %sum = linalg.generic {indexing_maps = [#id, #chan, #chan], iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%input, %max : tensor<?x?x?xf32>, tensor<?xf32>) outs(%init_sum : tensor<?xf32>)
      attrs = {lowering_config = #cfg} {
    ^bb0(%in: f32, %m: f32, %acc: f32):
      %d = arith.subf %in, %m : f32
      %e = math.exp %d : f32
      %s = arith.addf %e, %acc : f32
      linalg.yield %s : f32
  } -> tensor<?xf32>
  return %sum : tensor<?xf32>
}

// CHECK-LABEL: func.func @max_then_sum_two_reduction_dims
// CHECK-NOT: scf.for
// CHECK: %[[MAX:.*]] = linalg.generic
// CHECK: arith.maxnumf
// CHECK: scf.for
// CHECK: scf.for
// CHECK-NOT: arith.maxnumf
// CHECK: %[[MAX_SLICE:.*]] = tensor.extract_slice %[[MAX]][0]
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}}, %[[MAX_SLICE]] :
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: scf.yield
// CHECK-NEXT: }
// CHECK: scf.yield
// CHECK-NEXT: }
// CHECK-NOT: arith.maxnumf
// CHECK: return
