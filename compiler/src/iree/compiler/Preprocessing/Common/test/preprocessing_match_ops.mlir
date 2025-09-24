// RUN: iree-opt -transform-interpreter %s --split-input-file | FileCheck %s

#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @simple_max
func.func @simple_max(%input: tensor<?xf32>, %dest: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: linalg.generic
  // CHECK-SAME:   match_status = "matched"
  %res = linalg.generic {indexing_maps = [#map, #map],
                         iterator_types = ["parallel"]}
                         ins(%input : tensor<?xf32>)
                         outs(%dest : tensor<?xf32>) attrs = {match_status = "unmatched"} {
  ^bb0(%in: f32, %out: f32):
    %max = arith.maximumf %in, %out : f32
    linalg.yield %max : f32
  } -> tensor<?xf32>
  return %res : tensor<?xf32>
}

// CHECK: func.func @simple_min
func.func @simple_min(%input: tensor<?xf32>, %dest: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: linalg.generic
  // CHECK-SAME:   match_status = "unmatched"
  %res = linalg.generic {indexing_maps = [#map, #map],
                         iterator_types = ["parallel"]}
                         ins(%input : tensor<?xf32>)
                         outs(%dest : tensor<?xf32>) attrs = {match_status = "unmatched"} {
  ^bb0(%in: f32, %out: f32):
    %max = arith.minimumf %in, %out : f32
    linalg.yield %max : f32
  } -> tensor<?xf32>
  return %res : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %generic ["linalg.generic"] : !transform.any_op
    transform.iree.match.regions %generic : !transform.any_op {
      ^bb0(%target: tensor<f32>, %empty_max: tensor<f32>):
        %0 = linalg.generic {indexing_maps = [affine_map<() -> ()>,
                                                affine_map<() -> ()>],
                               iterator_types = []}
                               ins(%target : tensor<f32>)
                               outs(%empty_max : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %max = arith.maximumf %in, %out : f32
          linalg.yield %max : f32
        } -> tensor<f32>
    }
    transform.yield %generic : !transform.any_op
  }

  transform.named_sequence @annotate(%generic: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %generic "match_status" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach_match in %module
        @match -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

func.func private @external(%arg0: tensor<?xf32>)
func.func private @external_aligned(%arg0: tensor<100xf32>)
func.func private @external_static(%arg0: tensor<10xf32>)
func.func private @other_external_static(%arg0: tensor<15xf32>)
func.func private @external_2d(%arg0: tensor<?x?xf32>)

// CHECK-LABEL: func.func @call_external
func.func @call_external(%input: tensor<?xf32>,
                         %input_2d: tensor<?x?xf32>,
                         %input_aligned: tensor<100xf32>,
                         %input_static: tensor<10xf32>,
                         %other_static: tensor<15xf32>) {
//       CHECK: call @external
//  CHECK-SAME:   match_status = "matched"
  func.call @external(%input) {match_status = "unmatched"} : (tensor<?xf32>) -> ()
//       CHECK: call @external_2d
//  CHECK-SAME:   match_status = "unmatched"
  func.call @external_2d(%input_2d) {match_status = "unmatched"} : (tensor<?x?xf32>) -> ()
//       CHECK: call @external_aligned
//  CHECK-SAME:   match_status = "aligned_match"
  func.call @external_aligned(%input_aligned) {match_status = "unmatched"} : (tensor<100xf32>) -> ()
//       CHECK: call @external_static
//  CHECK-SAME:   match_status = "static_matched"
  func.call @external_static(%input_static) {match_status = "unmatched"} : (tensor<10xf32>) -> ()
//       CHECK: call @other_external_static
//  CHECK-SAME:   match_status = "matched"
  func.call @other_external_static(%other_static) {match_status = "unmatched"} : (tensor<15xf32>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @static_match(%call: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.match.operation_name %call ["func.call"] : !transform.any_op
    %in0 = transform.get_operand %call[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<10xf32> : !transform.any_value
    %0 = transform.param.constant "static_matched" -> !transform.any_param
    transform.yield %call, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @static_alignment_match(%call: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.match.operation_name %call ["func.call"] : !transform.any_op
    %in0 = transform.get_operand %call[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?xf32> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in0[0], 20 : !transform.any_value
    %0 = transform.param.constant "aligned_match" -> !transform.any_param
    transform.yield %call, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @match(%call: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.match.operation_name %call ["func.call"] : !transform.any_op
    %in0 = transform.get_operand %call[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<?xf32> : !transform.any_value
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.yield %call, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @annotate(%call: !transform.any_op {transform.readonly},
                                     %note: !transform.any_param {transform.readonly}) {
    transform.annotate %call "match_status" = %note : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach_match in %module
        @static_match -> @annotate,
        @static_alignment_match -> @annotate,
        @match -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

func.func private @external(%arg0: tensor<?xf32>)
func.func private @external_lb(%arg0: tensor<100xf32>)
func.func private @external_ub(%arg0: tensor<3xf32>)
func.func private @external_2d(%arg0: tensor<?x20xf32>)

// CHECK-LABEL: func.func @call_external
func.func @call_external(%arg0: index,
                         %input_2d: tensor<?x20xf32>,
                         %input_lb: tensor<100xf32>,
                         %input_ub: tensor<3xf32>) {
%0 = util.assume.int %arg0<umin = 12, umax = 16, udiv = 4> : index
%input = tensor.empty(%0) : tensor<?xf32>
//       CHECK: call @external
//  CHECK-SAME:   match_status = "both_matched"
  func.call @external(%input) {match_status = "unmatched"} : (tensor<?xf32>) -> ()
//       CHECK: call @external_2d
//  CHECK-SAME:   match_status = "dim1_matched"
  func.call @external_2d(%input_2d) {match_status = "unmatched"} : (tensor<?x20xf32>) -> ()
//       CHECK: call @external_lb
//  CHECK-SAME:   match_status = "lb_matched"
  func.call @external_lb(%input_lb) {match_status = "unmatched"} : (tensor<100xf32>) -> ()
//       CHECK: call @external_ub
//  CHECK-SAME:   match_status = "ub_matched"
  func.call @external_ub(%input_ub) {match_status = "unmatched"} : (tensor<3xf32>) -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @dim1_match(%call: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.match.operation_name %call ["func.call"] : !transform.any_op
    %in0 = transform.get_operand %call[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %in0[1], umin = 20, umax = 20 : !transform.any_value
    %0 = transform.param.constant "dim1_matched" -> !transform.any_param
    transform.yield %call, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @both_match(%call: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.match.operation_name %call ["func.call"] : !transform.any_op
    %in0 = transform.get_operand %call[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %in0[0], umin = 5, umax = 20 : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in0[0], 2 : !transform.any_value
    %0 = transform.param.constant "both_matched" -> !transform.any_param
    transform.yield %call, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @lb_match(%call: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.match.operation_name %call ["func.call"] : !transform.any_op
    %in0 = transform.get_operand %call[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %in0[0], umin = 75, none : !transform.any_value
    %0 = transform.param.constant "lb_matched" -> !transform.any_param
    transform.yield %call, %0 : !transform.any_op, !transform.any_param
  }
  transform.named_sequence @ub_match(%call: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    transform.match.operation_name %call ["func.call"] : !transform.any_op
    %in0 = transform.get_operand %call[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.dim_bounds %in0[0], none, umax = 4 : !transform.any_value
    %0 = transform.param.constant "ub_matched" -> !transform.any_param
    transform.yield %call, %0 : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @annotate(%call: !transform.any_op {transform.readonly},
                                     %note: !transform.any_param {transform.readonly}) {
    transform.annotate %call "match_status" = %note : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach_match in %module
        @dim1_match -> @annotate,
        @both_match -> @annotate,
        @lb_match -> @annotate,
        @ub_match -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {transform.with_named_sequence} {
  // CHECK: func.func @matmul_repeated_operand
  func.func @matmul_repeated_operand(%input: tensor<32x64xi8>, %dest: tensor<32x32xi32>) -> tensor<32x32xi32> {
    // CHECK-NEXT: linalg.matmul
    // CHECK-SAME:   match_status = "matched"
    %res = linalg.matmul
          indexing_maps = [#map0, #map1, #map2]
          {match_status = "unmatched"}
          ins(%input, %input : tensor<32x64xi8>, tensor<32x64xi8>)
          outs(%dest : tensor<32x32xi32>) -> tensor<32x32xi32>
    return %res : tensor<32x32xi32>
  }

  // CHECK: func.func @matmul_non_repeated_operand
  func.func @matmul_non_repeated_operand(%input0: tensor<32x64xi8>, %input1: tensor<32x64xi8>, %dest: tensor<32x32xi32>) -> tensor<32x32xi32> {
    // CHECK-NEXT: linalg.matmul
    // CHECK-SAME:   match_status = "unmatched"
    %res = linalg.matmul
          indexing_maps = [#map0, #map1, #map2]
          {match_status = "unmatched"}
          ins(%input0, %input1 : tensor<32x64xi8>, tensor<32x64xi8>)
          outs(%dest : tensor<32x32xi32>) -> tensor<32x32xi32>
    return %res : tensor<32x32xi32>
  }

  transform.named_sequence @match_matmul_repeated_operand(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %inputs, %outputs = transform.iree.match.cast_compatible_dag_from_root %arg0 {
    ^bb0(%arg1: tensor<32x64xi8>, %arg2: tensor<32x32xi32>):
      %1 = linalg.matmul
          indexing_maps = [#map0, #map1, #map2]
          {match_status = "unmatched"}
          ins(%arg1, %arg1 : tensor<32x64xi8>, tensor<32x64xi8>)
          outs(%arg2 : tensor<32x32xi32>) -> tensor<32x32xi32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @annotate(%generic: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %generic "match_status" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    %func = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.foreach_match in %module
        @match_matmul_repeated_operand -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// Verify that the basic contraction matcher works and can extract dimension sizes.

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_batch0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map_batch1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map_batch2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK-LABEL: func.func @op_matmul
func.func @op_matmul(%input0: tensor<32x64xi8>, %input1: tensor<32x64xi8>, %dest: tensor<32x32xi32>) -> tensor<32x32xi32> {
  // CHECK-NEXT: linalg.matmul
  // CHECK-SAME:   match_status = "matched"
  %res = linalg.matmul
        indexing_maps = [#map0, #map1, #map2]
        ins(%input0, %input1 : tensor<32x64xi8>, tensor<32x64xi8>)
        outs(%dest : tensor<32x32xi32>) {match_status = "unmatched"} -> tensor<32x32xi32>
  return %res : tensor<32x32xi32>
}

// CHECK-LABEL: func.func @op_batch_matmul
func.func @op_batch_matmul(%input0: tensor<2x32x64xi8>, %input1: tensor<2x32x64xi8>, %dest: tensor<2x32x32xi32>) -> tensor<2x32x32xi32> {
  // CHECK-NEXT: linalg.batch_matmul
  // CHECK-SAME:   match_status = "matched"
  %res = linalg.batch_matmul
        indexing_maps = [#map_batch0, #map_batch1, #map_batch2]
        ins(%input0, %input1 : tensor<2x32x64xi8>, tensor<2x32x64xi8>)
        outs(%dest : tensor<2x32x32xi32>) {match_status = "unmatched"} -> tensor<2x32x32xi32>
  return %res : tensor<2x32x32xi32>
}

// CHECK-LABEL: func.func @op_fill
func.func @op_fill(%dest: tensor<32x64xf32>, %value: f32) -> tensor<32x64xf32> {
  // CHECK-NEXT: linalg.fill
  // CHECK-SAME:   match_status = "unmatched"
  %res = linalg.fill ins(%value : f32) outs(%dest : tensor<32x64xf32>) {match_status = "unmatched"} -> tensor<32x64xf32>
  return %res : tensor<32x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_matmul(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %op,
      lhs_type = i8, rhs_type = i8, output_type = i32 :
      (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
    %c32 = transform.param.constant 32 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %m_dims, %c32 : !transform.param<i64>
    transform.match.param.cmpi eq %n_dims, %c32 : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @match_batch_matmul(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %op,
      lhs_type = i8, rhs_type = i8, output_type = i32 :
      (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
    %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
    %c32 = transform.param.constant 32 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %batch_dims, %c2 : !transform.param<i64>
    transform.match.param.cmpi eq %m_dims, %c32 : !transform.param<i64>
    transform.match.param.cmpi eq %n_dims, %c32 : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "match_status" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_matmul -> @annotate,
        @match_batch_matmul -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// Verify that contractions with exact same matching indexing maps are matched correctly,
// and contractions with different indexing map patterns are not matched.

#map_matmul0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_matmul1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_matmul2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_transpose_b = affine_map<(d0, d1, d2) -> (d2, d1)>  // Transpose_b RHS map.

// CHECK-LABEL: func.func @op_matmul
func.func @op_matmul(%input0: tensor<32x64xi8>, %input1: tensor<32x64xi8>, %input1_transposed: tensor<64x32xi8>, %dest: tensor<32x32xi32>) -> tensor<32x32xi32> {
  // CHECK-NEXT: linalg.matmul
  // CHECK-SAME:   indexing_maps_match = "matched"
  %res1 = linalg.matmul
        indexing_maps = [#map_matmul0, #map_matmul1, #map_matmul2]
        ins(%input0, %input1 : tensor<32x64xi8>, tensor<32x64xi8>)
        outs(%dest : tensor<32x32xi32>) {indexing_maps_match = "unmatched"} -> tensor<32x32xi32>

  // Transpose_b matmul - should NOT match (different RHS indexing map).
  // CHECK-NEXT: linalg.matmul
  // CHECK-SAME:   indexing_maps_match = "unmatched"
  %res2 = linalg.matmul
        indexing_maps = [#map_matmul0, #map_transpose_b, #map_matmul2]
        ins(%input0, %input1_transposed : tensor<32x64xi8>, tensor<64x32xi8>)
        outs(%dest : tensor<32x32xi32>) {indexing_maps_match = "unmatched"} -> tensor<32x32xi32>

  return %res1 : tensor<32x32xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_correct_maps(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
   %batch, %m, %n, %k = transform.iree.match.contraction %op,
    lhs_type = i8, rhs_type = i8, output_type = i32 {indexing_maps = [#map_matmul0, #map_matmul1, #map_matmul2]} :
    (!transform.any_op) ->
    (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate_matched(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "indexing_maps_match" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_correct_maps -> @annotate_matched
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// Verify that contractions with a different number of indexing maps are correctly not matched.

#map_matmul0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map_matmul1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_matmul2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @op_matmul
func.func @op_matmul(%input0: tensor<32x64xi8>, %input1: tensor<32x64xi8>, %dest: tensor<32x32xi32>) -> tensor<32x32xi32> {
  // CHECK-NEXT: linalg.matmul
  // CHECK-SAME:   indexing_maps_match = "unmatched"
  %res = linalg.matmul
        indexing_maps = [#map_matmul0, #map_matmul1, #map_matmul2]
        ins(%input0, %input1 : tensor<32x64xi8>, tensor<32x64xi8>)
        outs(%dest : tensor<32x32xi32>) {indexing_maps_match = "unmatched"} -> tensor<32x32xi32>
  return %res : tensor<32x32xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_different_count(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %m, %n, %k = transform.iree.match.contraction %op,
      lhs_type = i8, rhs_type = i8, output_type = i32 {indexing_maps = [#map_matmul0, #map_matmul1, #map_matmul2, #map_matmul0]} :
      (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate_matched(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "indexing_maps_match" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
     // Should NOT match: operation has 3 indexing maps but matcher expects 4.
    transform.foreach_match in %module
        @match_different_count -> @annotate_matched
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// Verify the contractions with lowering config can be matched successfully.

#promote_config = #iree_gpu.lowering_config<{promote_operands = [0, 1]}>
#mma_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>}>

// CHECK-LABEL: func.func @test_promote_operands_config
func.func @test_promote_operands_config(%a: tensor<32x1024xf32>, %b: tensor<1024x128xf32>, %c: tensor<32x128xf32>) -> tensor<32x128xf32> {
  // CHECK-NEXT: linalg.matmul
  // CHECK-SAME:   lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1]}>
  // CHECK-SAME:   match_status = "matched_with_lowering_config"
  %mm = linalg.matmul {lowering_config = #promote_config}
    ins(%a, %b : tensor<32x1024xf32>, tensor<1024x128xf32>)
    outs(%c : tensor<32x128xf32>) -> tensor<32x128xf32>
  return %mm : tensor<32x128xf32>
}

// CHECK-LABEL: func.func @test_mma_layout_config
func.func @test_mma_layout_config(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  // CHECK-NEXT: linalg.matmul
  // CHECK-SAME:   lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>}>
  // CHECK-SAME:   match_status = "matched_with_lowering_config"
  %mm = linalg.matmul {lowering_config = #mma_config}
    ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %mm : tensor<64x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_matmul(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %op,
      lhs_type = f32, rhs_type = f32, output_type = f32 :
      (!transform.any_op) -> (!transform.param<i64>, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched_with_lowering_config" -> !transform.any_param
    transform.annotate %op "match_status" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_matmul -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
