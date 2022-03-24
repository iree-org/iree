// RUN: iree-dialects-opt -linalg-interp-transforms %s | FileCheck %s

//       CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func @interchange_generic
func @interchange_generic(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {

  //      CHECK:   linalg.generic
  // CHECK-SAME:   indexing_maps = [#[[$MAP]], #[[$MAP]]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %1 = math.exp %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}


pdl.pattern @pdl_target : benefit(1) {
  %args = operands
  %results = types
  %0 = pdl.operation "linalg.generic"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@interchange_generic](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @pdl_target
  interchange %0 {iterator_interchange = [1, 0]}
}
