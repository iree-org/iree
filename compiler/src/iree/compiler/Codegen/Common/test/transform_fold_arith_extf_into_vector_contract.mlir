// RUN: iree-opt -iree-transform-dialect-interpreter %s | FileCheck %s


// CHECK-DAG: #[[$map0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$map1:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$map2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @fold_arith_extf_into_contract
//  CHECK-SAME: (%[[ARG0:.*]]: vector<64x64xf16>, %[[ARG1:.*]]: vector<64x64xf16>, %[[ARG2:.*]]: vector<64x64xf32>)
//  CHECK-NEXT:   %[[R:.+]] = vector.contract {indexing_maps = [#[[$map0]], #[[$map1]], #[[$map2]]],
//  CHECK-SAME:   iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>}
//  CHECK-SAME:   %[[ARG0]], %[[ARG1]], %[[ARG2]] : vector<64x64xf16>, vector<64x64xf16> into vector<64x64xf32>
//  CHECK-NEXT:   return %[[R]] : vector<64x64xf32>
func.func @fold_arith_extf_into_contract(%arg0: vector<64x64xf16>, %arg1: vector<64x64xf16>, %arg2: vector<64x64xf32>) -> vector<64x64xf32> {
    %lhs_f32 = arith.extf %arg0 : vector<64x64xf16> to vector<64x64xf32>
    %rhs_f32 = arith.extf %arg1 : vector<64x64xf16> to vector<64x64xf32>
    %result = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs_f32, %rhs_f32, %arg2 : vector<64x64xf32>, vector<64x64xf32> into vector<64x64xf32>
    return %result : vector<64x64xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %0 {
      transform.apply_patterns.iree.fold_arith_ext_into_contraction
    } : !transform.any_op
    transform.yield
  } // @__transform_main
} // module
