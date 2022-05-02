// RUN: iree-dialects-opt -linalg-transform-interp %s | FileCheck %s

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

transform.with_pdl_patterns {
^bb0(%root: !pdl.operation):
  pdl.pattern @pdl_target : benefit(1) {
    %args = operands
    %results = types
    %0 = pdl.operation "linalg.generic"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    %1 = pdl.attribute @interchange_generic
    apply_native_constraint "nestedInFunc"(%0, %1 : !pdl.operation, !pdl.attribute)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %root {
  ^bb0(%arg0: !pdl.operation):
    %0 = pdl_match @pdl_target in %arg0
    transform.structured.interchange %0 {iterator_interchange = [1, 0]}
  }
}
