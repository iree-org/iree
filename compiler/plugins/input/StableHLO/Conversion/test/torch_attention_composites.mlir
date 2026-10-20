// RUN: iree-opt --split-input-file --iree-stablehlo-input-transformation-pipeline %s | FileCheck %s

// CHECK-DAG:   #[[$Q:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
// CHECK-DAG:   #[[$K:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
// CHECK-DAG:   #[[$V:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d3)>
// CHECK-DAG:   #[[$S:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> ()>
// CHECK-DAG:   #[[$O:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @sdpa(
//  CHECK-SAME:     %[[Q:[a-z0-9]+]]: tensor<2x3x5x4xf32>, %[[K:[a-z0-9]+]]: tensor<2x3x7x4xf32>, %[[V:[a-z0-9]+]]: tensor<2x3x7x6xf32>
//       CHECK:   %[[SCALE:.+]] = arith.constant 5.000000e-01 : f32
//       CHECK:   iree_linalg_ext.attention
//  CHECK-SAME:     indexing_maps = [#[[$Q]], #[[$K]], #[[$V]], #[[$S]], #[[$O]]]
//  CHECK-SAME:     ins(%[[Q]], %[[K]], %[[V]], %[[SCALE]] :
//  CHECK-SAME:     outs(%{{.+}} : tensor<2x3x5x6xf32>)
//   CHECK-NOT:   linalg.softmax
func.func @sdpa(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_impl, version = 1 : i32
  } : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>, tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32>
  return %0 : tensor<2x3x5x6xf32>
}
func.func private @sdpa_impl(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>) -> tensor<2x3x5x7xf32>
  %1 = stablehlo.dot_general %0, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x3x5x7xf32>, tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32>
  return %1 : tensor<2x3x5x6xf32>
}

// -----

// A causal composite hides keys after the query in the region, from the query
// and key score indices.
// CHECK-LABEL: func.func @sdpa_causal_scale(
//       CHECK:   %[[SCALE:.+]] = arith.constant 2.500000e-01 : f32
//   CHECK-NOT:   arith.cmpi
//       CHECK:   iree_linalg_ext.attention
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}}, %{{.+}}, %[[SCALE]] : tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>, tensor<2x3x7x6xf32>, f32)
//       CHECK:     ^bb0(%[[SCORE:.+]]: f32):
//       CHECK:       %[[QUERY:.+]] = iree_linalg_ext.index 2 : index
//       CHECK:       %[[KEY:.+]] = iree_linalg_ext.index 3 : index
//       CHECK:       %[[HIDDEN:.+]] = arith.cmpi ugt, %[[KEY]], %[[QUERY]] : index
//       CHECK:       %[[MASKED:.+]] = arith.select %[[HIDDEN]], %{{.+}}, %[[SCORE]] : f32
//       CHECK:       iree_linalg_ext.yield %[[MASKED]] : f32
func.func @sdpa_causal_scale(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = true, scale = 2.500000e-01 : f32},
    decomposition = @sdpa_causal_impl, version = 1 : i32
  } : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>, tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32>
  return %0 : tensor<2x3x5x6xf32>
}
func.func private @sdpa_causal_impl(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>) -> tensor<2x3x5x7xf32>
  %1 = stablehlo.dot_general %0, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x3x5x7xf32>, tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32>
  return %1 : tensor<2x3x5x6xf32>
}


// -----

// Dynamic query and key lengths need no mask tensor shape.
// CHECK-LABEL: func.func @sdpa_causal_dynamic(
//       CHECK:   iree_linalg_ext.attention
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<2x3x?x4xf32>, tensor<2x3x?x4xf32>, tensor<2x3x?x6xf32>, f32)
//       CHECK:       iree_linalg_ext.index 2 : index
//       CHECK:       iree_linalg_ext.index 3 : index
func.func @sdpa_causal_dynamic(%q: tensor<2x3x?x4xf32>, %k: tensor<2x3x?x4xf32>, %v: tensor<2x3x?x6xf32>) -> tensor<2x3x?x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = true},
    decomposition = @sdpa_causal_dynamic_impl, version = 1 : i32
  } : (tensor<2x3x?x4xf32>, tensor<2x3x?x4xf32>, tensor<2x3x?x6xf32>) -> tensor<2x3x?x6xf32>
  return %0 : tensor<2x3x?x6xf32>
}
func.func private @sdpa_causal_dynamic_impl(%q: tensor<2x3x?x4xf32>, %k: tensor<2x3x?x4xf32>, %v: tensor<2x3x?x6xf32>) -> tensor<2x3x?x6xf32> {
  %0 = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<2x3x?x4xf32>, tensor<2x3x?x4xf32>) -> tensor<2x3x?x?xf32>
  %1 = stablehlo.dot_general %0, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x3x?x?xf32>, tensor<2x3x?x6xf32>) -> tensor<2x3x?x6xf32>
  return %1 : tensor<2x3x?x6xf32>
}
// -----

// A per-batch additive mask maps over every batch dimension. A row whose mask
// is -inf everywhere is zero, as torch's _safe_softmax makes it.
// CHECK-DAG:   #[[$BATCH_MASK:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>
// CHECK-LABEL: func.func @sdpa_float_mask(
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:     #[[$BATCH_MASK]]
//  CHECK-SAME:     tensor<2x3x5x7xf32>)
//       CHECK:   %[[VISIBLE:.+]] = linalg.generic
//       CHECK:     arith.cmpf une, %{{.+}}, %cst{{.*}} : f32
//       CHECK:     arith.ori
//       CHECK:     -> tensor<2x3x5xi1>
//       CHECK:   linalg.generic {{.+}} ins(%[[ATTENTION]], %[[VISIBLE]] : tensor<2x3x5x6xf32>, tensor<2x3x5xi1>)
//       CHECK:     arith.select
func.func @sdpa_float_mask(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>, %m: tensor<2x3x5x7xf32>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v, %m {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_float_mask_impl, version = 1 : i32
  } : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>, tensor<2x3x7x6xf32>, tensor<2x3x5x7xf32>) -> tensor<2x3x5x6xf32>
  return %0 : tensor<2x3x5x6xf32>
}
func.func private @sdpa_float_mask_impl(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>, %m: tensor<2x3x5x7xf32>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>) -> tensor<2x3x5x7xf32>
  %1 = stablehlo.dot_general %0, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x3x5x7xf32>, tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32>
  return %1 : tensor<2x3x5x6xf32>
}

// -----

// A mask broadcast over heads drops its unit dimension and maps over the
// remaining batch dimension only.
// CHECK-DAG:   #[[$HEAD_BROADCAST_MASK:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d5)>
// CHECK-LABEL: func.func @sdpa_head_broadcast_mask(
//       CHECK:   %[[MASK:.+]] = tensor.collapse_shape %{{.+}} {{\[}}[0], [1, 2], [3]] : tensor<2x1x5x7xf32> into tensor<2x5x7xf32>
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:     #[[$HEAD_BROADCAST_MASK]]
//  CHECK-SAME:     tensor<2x5x7xf32>)
//       CHECK:   %[[VISIBLE:.+]] = linalg.generic
//       CHECK:     -> tensor<2x5xi1>
//       CHECK:   linalg.generic {{.+}} ins(%[[ATTENTION]], %[[VISIBLE]] : tensor<2x3x5x6xf32>, tensor<2x5xi1>)
func.func @sdpa_head_broadcast_mask(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>, %m: tensor<2x1x5x7xf32>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v, %m {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_head_broadcast_mask_impl, version = 1 : i32
  } : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>, tensor<2x3x7x6xf32>, tensor<2x1x5x7xf32>) -> tensor<2x3x5x6xf32>
  return %0 : tensor<2x3x5x6xf32>
}
func.func private @sdpa_head_broadcast_mask_impl(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>, %m: tensor<2x1x5x7xf32>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<2x3x5x6xf32>
  return %0 : tensor<2x3x5x6xf32>
}

// -----

// A constant mask with a visible key in every row needs no hidden-row fixup.
// CHECK-LABEL: func.func @sdpa_constant_visible_mask(
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//   CHECK-NOT:   arith.select
//       CHECK:   return %[[ATTENTION]]
func.func @sdpa_constant_visible_mask(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32> {
  %m = stablehlo.constant dense<[[true, false, false, false, false, false, false],
                                 [true, true, false, false, false, false, false],
                                 [false, false, true, false, false, false, false],
                                 [false, false, false, true, true, false, false],
                                 [false, false, false, false, false, false, true]]> : tensor<5x7xi1>
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v, %m {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_constant_visible_mask_impl, version = 1 : i32
  } : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>, tensor<2x3x7x6xf32>, tensor<5x7xi1>) -> tensor<2x3x5x6xf32>
  return %0 : tensor<2x3x5x6xf32>
}
func.func private @sdpa_constant_visible_mask_impl(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>, %m: tensor<5x7xi1>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<2x3x5x6xf32>
  return %0 : tensor<2x3x5x6xf32>
}

// -----

// A constant mask that hides a whole row keeps the fixup.
// CHECK-LABEL: func.func @sdpa_constant_hidden_row_mask(
//       CHECK:   iree_linalg_ext.attention
//       CHECK:   arith.select
func.func @sdpa_constant_hidden_row_mask(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x3x5x6xf32> {
  %m = stablehlo.constant dense<[[true, false, false, false, false, false, false],
                                 [false, false, false, false, false, false, false],
                                 [false, false, true, false, false, false, false],
                                 [false, false, false, true, true, false, false],
                                 [false, false, false, false, false, false, true]]> : tensor<5x7xi1>
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v, %m {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_constant_hidden_row_mask_impl, version = 1 : i32
  } : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>, tensor<2x3x7x6xf32>, tensor<5x7xi1>) -> tensor<2x3x5x6xf32>
  return %0 : tensor<2x3x5x6xf32>
}
func.func private @sdpa_constant_hidden_row_mask_impl(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>, %m: tensor<5x7xi1>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<2x3x5x6xf32>
  return %0 : tensor<2x3x5x6xf32>
}

// -----

// Six query heads share three key and value heads: the query splits its heads
// into (3, 2) groups and K/V are not repeated.
// CHECK-DAG:   #[[$GQ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d5)>
// CHECK-DAG:   #[[$GK:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d6, d5)>
// CHECK-DAG:   #[[$GV:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d6, d4)>
// CHECK-DAG:   #[[$GO:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @sdpa_gqa(
//  CHECK-SAME:     %[[Q:[a-z0-9]+]]: tensor<2x6x5x4xf32>, %[[K:[a-z0-9]+]]: tensor<2x3x7x4xf32>, %[[V:[a-z0-9]+]]: tensor<2x3x7x6xf32>
//       CHECK:   %[[GROUPED:.+]] = tensor.expand_shape %[[Q]] {{\[}}[0], [1, 2], [3], [4]] output_shape [2, 3, 2, 5, 4] : tensor<2x6x5x4xf32> into tensor<2x3x2x5x4xf32>
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:     indexing_maps = [#[[$GQ]], #[[$GK]], #[[$GV]], #{{.+}}, #[[$GO]]]
//  CHECK-SAME:     ins(%[[GROUPED]], %[[K]], %[[V]], %{{[a-z0-9_]+}} :
//  CHECK-SAME:     outs(%{{.+}} : tensor<2x3x2x5x6xf32>)
//       CHECK:   tensor.collapse_shape %[[ATTENTION]] {{\[}}[0], [1, 2], [3], [4]] : tensor<2x3x2x5x6xf32> into tensor<2x6x5x6xf32>
func.func @sdpa_gqa(%q: tensor<2x6x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x6x5x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = true, is_causal = false},
    decomposition = @sdpa_gqa_impl, version = 1 : i32
  } : (tensor<2x6x5x4xf32>, tensor<2x3x7x4xf32>, tensor<2x3x7x6xf32>) -> tensor<2x6x5x6xf32>
  return %0 : tensor<2x6x5x6xf32>
}
func.func private @sdpa_gqa_impl(%q: tensor<2x6x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x6x5x6xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<2x6x5x6xf32>
  return %0 : tensor<2x6x5x6xf32>
}

// -----

// A per-head mask splits its heads like the query and keeps the query and key
// positions; lengths are dynamic.
// CHECK-DAG:   #[[$GQ:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d5)>
// CHECK-DAG:   #[[$GM:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d6)>
// CHECK-DAG:   #[[$GO:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @sdpa_gqa_projected_maps(
//  CHECK-SAME:     %[[Q:[a-z0-9]+]]: tensor<2x4x?x4xf32>, %[[K:[a-z0-9]+]]: tensor<2x2x?x4xf32>, %[[V:[a-z0-9]+]]: tensor<2x2x?x6xf32>, %[[M:[a-z0-9]+]]: tensor<2x4x?x?xf32>
//   CHECK-DAG:   %[[GROUPED:.+]] = tensor.expand_shape %[[Q]] {{\[}}[0], [1, 2], [3], [4]] output_shape [2, 2, 2, %{{.+}}, 4] : tensor<2x4x?x4xf32> into tensor<2x2x2x?x4xf32>
//   CHECK-DAG:   %[[GMASK:.+]] = tensor.expand_shape %[[M]] {{\[}}[0], [1, 2], [3], [4]] output_shape [2, 2, 2, %{{.+}}, %{{.+}}] : tensor<2x4x?x?xf32> into tensor<2x2x2x?x?xf32>
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:     indexing_maps = [#[[$GQ]], #{{.+}}, #{{.+}}, #{{.+}}, #[[$GM]], #[[$GO]]]
//  CHECK-SAME:     ins(%[[GROUPED]], %[[K]], %[[V]], %{{[a-z0-9_]+}}, %[[GMASK]] :
//       CHECK:   tensor.collapse_shape %[[ATTENTION]] {{\[}}[0], [1, 2], [3], [4]] : tensor<2x2x2x?x6xf32> into tensor<2x4x?x6xf32>
func.func @sdpa_gqa_projected_maps(%q: tensor<2x4x?x4xf32>, %k: tensor<2x2x?x4xf32>, %v: tensor<2x2x?x6xf32>, %mask: tensor<2x4x?x?xf32>) -> tensor<2x4x?x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v, %mask {
    composite_attributes = {enable_gqa = true, is_causal = false},
    decomposition = @sdpa_gqa_projected_maps_impl, version = 1 : i32
  } : (tensor<2x4x?x4xf32>, tensor<2x2x?x4xf32>, tensor<2x2x?x6xf32>, tensor<2x4x?x?xf32>) -> tensor<2x4x?x6xf32>
  return %0 : tensor<2x4x?x6xf32>
}
func.func private @sdpa_gqa_projected_maps_impl(%q: tensor<2x4x?x4xf32>, %k: tensor<2x2x?x4xf32>, %v: tensor<2x2x?x6xf32>, %mask: tensor<2x4x?x?xf32>) -> tensor<2x4x?x6xf32> {
  %w = stablehlo.constant dense<0.0> : tensor<4x6xf32>
  %0 = stablehlo.dot_general %q, %w, contracting_dims = [3] x [0] : (tensor<2x4x?x4xf32>, tensor<4x6xf32>) -> tensor<2x4x?x6xf32>
  return %0 : tensor<2x4x?x6xf32>
}

// -----

// A causal grouped-query composite hides keys after the query by the query
// and key positions, score indices 3 and 4 after the head and group.
// CHECK-LABEL: func.func @sdpa_gqa_causal(
//       CHECK:   iree_linalg_ext.attention
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<2x2x2x?x4xf32>, tensor<2x2x?x4xf32>, tensor<2x2x?x6xf32>, f32)
//       CHECK:       %[[QUERY:.+]] = iree_linalg_ext.index 3 : index
//       CHECK:       %[[KEY:.+]] = iree_linalg_ext.index 4 : index
//       CHECK:       arith.cmpi ugt, %[[KEY]], %[[QUERY]] : index
func.func @sdpa_gqa_causal(%q: tensor<2x4x?x4xf32>, %k: tensor<2x2x?x4xf32>, %v: tensor<2x2x?x6xf32>) -> tensor<2x4x?x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = true, is_causal = true},
    decomposition = @sdpa_gqa_causal_impl, version = 1 : i32
  } : (tensor<2x4x?x4xf32>, tensor<2x2x?x4xf32>, tensor<2x2x?x6xf32>) -> tensor<2x4x?x6xf32>
  return %0 : tensor<2x4x?x6xf32>
}
func.func private @sdpa_gqa_causal_impl(%q: tensor<2x4x?x4xf32>, %k: tensor<2x2x?x4xf32>, %v: tensor<2x2x?x6xf32>) -> tensor<2x4x?x6xf32> {
  %w = stablehlo.constant dense<0.0> : tensor<4x6xf32>
  %0 = stablehlo.dot_general %q, %w, contracting_dims = [3] x [0] : (tensor<2x4x?x4xf32>, tensor<4x6xf32>) -> tensor<2x4x?x6xf32>
  return %0 : tensor<2x4x?x6xf32>
}

// -----

// One key and value head serves every query head.
// CHECK-LABEL: func.func @sdpa_mqa(
//       CHECK:   tensor.expand_shape %{{.+}} {{\[}}[0], [1, 2], [3], [4]] output_shape [2, 1, 4, 5, 4] : tensor<2x4x5x4xf32> into tensor<2x1x4x5x4xf32>
//       CHECK:   iree_linalg_ext.attention
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<2x1x4x5x4xf32>, tensor<2x1x7x4xf32>, tensor<2x1x7x6xf32>, f32)
func.func @sdpa_mqa(%q: tensor<2x4x5x4xf32>, %k: tensor<2x1x7x4xf32>, %v: tensor<2x1x7x6xf32>) -> tensor<2x4x5x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = true, is_causal = false},
    decomposition = @sdpa_mqa_impl, version = 1 : i32
  } : (tensor<2x4x5x4xf32>, tensor<2x1x7x4xf32>, tensor<2x1x7x6xf32>) -> tensor<2x4x5x6xf32>
  return %0 : tensor<2x4x5x6xf32>
}
func.func private @sdpa_mqa_impl(%q: tensor<2x4x5x4xf32>, %k: tensor<2x1x7x4xf32>, %v: tensor<2x1x7x6xf32>) -> tensor<2x4x5x6xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<2x4x5x6xf32>
  return %0 : tensor<2x4x5x6xf32>
}

// -----

// Six query heads cannot share four key heads: the decomposition stays.
// CHECK-LABEL: func.func @sdpa_gqa_indivisible(
//   CHECK-NOT:   iree_linalg_ext.attention
//       CHECK:   return
func.func @sdpa_gqa_indivisible(%q: tensor<2x6x5x4xf32>, %k: tensor<2x4x7x4xf32>, %v: tensor<2x4x7x6xf32>) -> tensor<2x6x5x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = true, is_causal = false},
    decomposition = @sdpa_gqa_indivisible_impl, version = 1 : i32
  } : (tensor<2x6x5x4xf32>, tensor<2x4x7x4xf32>, tensor<2x4x7x6xf32>) -> tensor<2x6x5x6xf32>
  return %0 : tensor<2x6x5x6xf32>
}
func.func private @sdpa_gqa_indivisible_impl(%q: tensor<2x6x5x4xf32>, %k: tensor<2x4x7x4xf32>, %v: tensor<2x4x7x6xf32>) -> tensor<2x6x5x6xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<2x6x5x6xf32>
  return %0 : tensor<2x6x5x6xf32>
}

// -----

// Two run-time head counts may differ: the decomposition stays.
// CHECK-LABEL: func.func @sdpa_gqa_both_dynamic_heads(
//   CHECK-NOT:   iree_linalg_ext.attention
//       CHECK:   return
func.func @sdpa_gqa_both_dynamic_heads(%q: tensor<2x?x5x4xf32>, %k: tensor<2x?x7x4xf32>, %v: tensor<2x?x7x6xf32>) -> tensor<2x?x5x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = true, is_causal = false},
    decomposition = @sdpa_gqa_both_dynamic_heads_impl, version = 1 : i32
  } : (tensor<2x?x5x4xf32>, tensor<2x?x7x4xf32>, tensor<2x?x7x6xf32>) -> tensor<2x?x5x6xf32>
  return %0 : tensor<2x?x5x6xf32>
}
func.func private @sdpa_gqa_both_dynamic_heads_impl(%q: tensor<2x?x5x4xf32>, %k: tensor<2x?x7x4xf32>, %v: tensor<2x?x7x6xf32>) -> tensor<2x?x5x6xf32> {
  %w = stablehlo.constant dense<0.0> : tensor<4x6xf32>
  %0 = stablehlo.dot_general %q, %w, contracting_dims = [3] x [0] : (tensor<2x?x5x4xf32>, tensor<4x6xf32>) -> tensor<2x?x5x6xf32>
  return %0 : tensor<2x?x5x6xf32>
}

// -----

// A run-time head count proves no grouping: the decomposition stays.
// CHECK-LABEL: func.func @sdpa_gqa_dynamic_heads(
//   CHECK-NOT:   iree_linalg_ext.attention
//       CHECK:   return
func.func @sdpa_gqa_dynamic_heads(%q: tensor<2x?x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x?x5x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = true, is_causal = false},
    decomposition = @sdpa_gqa_dynamic_heads_impl, version = 1 : i32
  } : (tensor<2x?x5x4xf32>, tensor<2x3x7x4xf32>, tensor<2x3x7x6xf32>) -> tensor<2x?x5x6xf32>
  return %0 : tensor<2x?x5x6xf32>
}
func.func private @sdpa_gqa_dynamic_heads_impl(%q: tensor<2x?x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>) -> tensor<2x?x5x6xf32> {
  %w = stablehlo.constant dense<0.0> : tensor<4x6xf32>
  %0 = stablehlo.dot_general %q, %w, contracting_dims = [3] x [0] : (tensor<2x?x5x4xf32>, tensor<4x6xf32>) -> tensor<2x?x5x6xf32>
  return %0 : tensor<2x?x5x6xf32>
}

// -----

// A shared boolean mask: a row with no true entry is zero.
// CHECK-LABEL: func.func @sdpa_bool_mask(
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//       CHECK:   %[[VISIBLE:.+]] = linalg.generic
//   CHECK-NOT:     arith.cmpf
//       CHECK:     arith.ori
//       CHECK:     -> tensor<5xi1>
//       CHECK:   linalg.generic {{.+}} ins(%[[ATTENTION]], %[[VISIBLE]] : tensor<2x3x5x6xf32>, tensor<5xi1>)
//       CHECK:     arith.select
func.func @sdpa_bool_mask(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>, %m: tensor<5x7xi1>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v, %m {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_bool_mask_impl, version = 1 : i32
  } : (tensor<2x3x5x4xf32>, tensor<2x3x7x4xf32>, tensor<2x3x7x6xf32>, tensor<5x7xi1>) -> tensor<2x3x5x6xf32>
  return %0 : tensor<2x3x5x6xf32>
}
func.func private @sdpa_bool_mask_impl(%q: tensor<2x3x5x4xf32>, %k: tensor<2x3x7x4xf32>, %v: tensor<2x3x7x6xf32>, %m: tensor<5x7xi1>) -> tensor<2x3x5x6xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<2x3x5x6xf32>
  return %0 : tensor<2x3x5x6xf32>
}

// -----

// The attention op accumulates in f32, so f64 keeps the decomposition.
// CHECK-LABEL: func.func @sdpa_f64(
//   CHECK-NOT:   iree_linalg_ext.attention
//       CHECK:   return
func.func @sdpa_f64(%q: tensor<2x3x5x4xf64>, %k: tensor<2x3x7x4xf64>, %v: tensor<2x3x7x6xf64>) -> tensor<2x3x5x6xf64> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_f64_impl, version = 1 : i32
  } : (tensor<2x3x5x4xf64>, tensor<2x3x7x4xf64>, tensor<2x3x7x6xf64>) -> tensor<2x3x5x6xf64>
  return %0 : tensor<2x3x5x6xf64>
}
func.func private @sdpa_f64_impl(%q: tensor<2x3x5x4xf64>, %k: tensor<2x3x7x4xf64>, %v: tensor<2x3x7x6xf64>) -> tensor<2x3x5x6xf64> {
  %0 = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<2x3x5x4xf64>, tensor<2x3x7x4xf64>) -> tensor<2x3x5x7xf64>
  %1 = stablehlo.dot_general %0, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x3x5x7xf64>, tensor<2x3x7x6xf64>) -> tensor<2x3x5x6xf64>
  return %1 : tensor<2x3x5x6xf64>
}
