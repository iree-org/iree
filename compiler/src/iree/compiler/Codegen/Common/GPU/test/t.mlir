// RUN: iree-opt -print-local-scope --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-apply-tiling-level{tiling-level=partial_reduction}, canonicalize, cse))" %s | FileCheck %s --check-prefix=PARTRED

#config = #iree_gpu.lowering_config<{partial_reduction = [0, 8]}>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @partial_reduction(%3: tensor<?x?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0  = arith.constant 0 : index
  %par_dim = tensor.dim %3, %c0 : tensor<?x?xf32>
  %empty = tensor.empty(%par_dim) : tensor<?xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
  %5 = linalg.generic {
    indexing_maps = [#map1, #map2],
    iterator_types = ["parallel", "reduction"]
    } ins(%3 : tensor<?x?xf32>) outs(%4 : tensor<?xf32>) attrs =  {lowering_config = #config} {
  ^bb0(%in: f32, %out: f32):
    %7 = arith.addf %in, %out : f32
    linalg.yield %7 : f32
  } -> tensor<?xf32>
  return %5 : tensor<?xf32>
}
