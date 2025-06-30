// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-cpu-tile-and-distribute-to-workgroups, cse))" --mlir-print-local-scope --split-input-file %s | FileCheck %s

#config1 = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 0, 0, 0, 0], [1, 1, 0, 16, 16, 0], [0, 0, 1, 0, 0, 1]]>
#config2 = #iree_codegen.lowering_config<tile_sizes = [[0, 0], [0, 0], [0, 0], [0, 0]]>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @mmt4d_unpack_elementwise(
    %lhs: tensor<?x?x16x1xf32>, %rhs: tensor<?x?x16x1xf32>, %acc: tensor<?x?x16x16xf32>,
    %unpack_dest: tensor<?x?xf32>, %elementwise_input: tensor<?xf32>,
    %dest: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.mmt4d {lowering_config = #config1} ins(%lhs, %rhs : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>) outs(%acc : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  %unpack = linalg.unpack %0
    outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16]
    into %unpack_dest {lowering_config = #config2}
    : tensor<?x?x16x16xf32> -> tensor<?x?xf32>
  %1 = linalg.generic {
    indexing_maps = [#map1, #map2, #map1],
    iterator_types = ["parallel", "parallel"]
  } ins(%unpack, %elementwise_input : tensor<?x?xf32>, tensor<?xf32>)
    outs(%dest : tensor<?x?xf32>)
    attrs = {lowering_config = #config2} {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %2 = arith.addf %in, %in_0 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
// If the mmt4d op is selected as a root op, scf.forall is generated. Otherwise,
// it is a nop pass because the other config uses zero values. It checks the
// in_parallel op at the end which ensures that all the ops are fused into the
// forall op.
// CHECK-LABEL: @mmt4d_unpack_elementwise(
// CHECK:         scf.forall
// CHECK:           linalg.mmt4d
// CHECK:           linalg.unpack
// CHECK:           linalg.generic
// CHECK:         scf.forall.in_parallel
