// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-and-fuse-producer-consumer{tiling-level=vector_reduction only-fuse-producer-input-operands=true}))" --split-input-file %s | FileCheck %s

// Keep the row maximum outside the reduction loop, even without LICM.
// The exponential and sum must stay inside the key-tile loop.
#id = affine_map<(r, k) -> (r, k)>
#row = affine_map<(r, k) -> (r)>
#cfg = #iree_cpu.lowering_config<distribution = [32, 0], vector_common_parallel = [4, 0], vector_reduction = [0, 4]>
func.func @max_then_sum(%scores: tensor<?x?xf32>) -> tensor<?xf32> {
  %neg = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %rows = tensor.dim %scores, %c0 : tensor<?x?xf32>
  %c1 = arith.constant 1 : index
  %keys_raw = tensor.dim %scores, %c1 : tensor<?x?xf32>
  %keys = util.assume.int %keys_raw<umin = 32, umax = 4096, udiv = 32> : index
  %bounded = flow.tensor.tie_shape %scores : tensor<?x?xf32>{%rows, %keys}
  %empty = tensor.empty(%rows) : tensor<?xf32>
  %init_max = linalg.fill ins(%neg : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %max = linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]}
      ins(%bounded : tensor<?x?xf32>) outs(%init_max : tensor<?xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %m = arith.maxnumf %x, %acc : f32
      linalg.yield %m : f32
  } -> tensor<?xf32>
  %init_sum = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %sum = linalg.generic {indexing_maps = [#id, #row, #row], iterator_types = ["parallel", "reduction"]}
      ins(%bounded, %max : tensor<?x?xf32>, tensor<?xf32>) outs(%init_sum : tensor<?xf32>)
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
// CHECK: return
