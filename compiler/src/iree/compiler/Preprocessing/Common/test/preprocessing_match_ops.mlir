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
      lhs_type = i8, rhs_type = i8, output_type = i32 : !transform.any_op -> !transform.param<i64>
    %c32 = transform.param.constant 32 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %m_dims, %c32 : !transform.param<i64>
    transform.match.param.cmpi eq %n_dims, %c32 : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @match_batch_matmul(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %op,
      lhs_type = i8, rhs_type = i8, output_type = i32 : !transform.any_op -> !transform.param<i64>
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
    lhs_type = i8, rhs_type = i8, output_type = i32, indexing_maps = [#map_matmul0, #map_matmul1, #map_matmul2] :
    !transform.any_op -> !transform.param<i64>
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
      lhs_type = i8, rhs_type = i8, output_type = i32, indexing_maps = [#map_matmul0, #map_matmul1, #map_matmul2, #map_matmul0] :
      !transform.any_op -> !transform.param<i64>
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
      lhs_type = f32, rhs_type = f32, output_type = f32 : !transform.any_op -> !transform.param<i64>
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

// -----

// Verify MatchSizeEqualsOp with dimension size constraints.

// CHECK-LABEL: func.func @op_matmul_4096x2048x8192
func.func @op_matmul_4096x2048x8192(%input0: tensor<4096x8192xf16>, %input1: tensor<8192x2048xf16>, %dest: tensor<4096x2048xf32>) -> tensor<4096x2048xf32> {
  // CHECK-NEXT: linalg.matmul
  // CHECK-SAME:   match_status = "matched"
  %res = linalg.matmul
        ins(%input0, %input1 : tensor<4096x8192xf16>, tensor<8192x2048xf16>)
        outs(%dest : tensor<4096x2048xf32>) {match_status = "unmatched"} -> tensor<4096x2048xf32>
  return %res : tensor<4096x2048xf32>
}

// CHECK-LABEL: func.func @op_matmul_1024x512x2048
func.func @op_matmul_1024x512x2048(%input0: tensor<1024x2048xf16>, %input1: tensor<2048x512xf16>, %dest: tensor<1024x512xf32>) -> tensor<1024x512xf32> {
  // CHECK-NEXT: linalg.matmul
  // CHECK-SAME:   match_status = "unmatched"
  %res = linalg.matmul
        ins(%input0, %input1 : tensor<1024x2048xf16>, tensor<2048x512xf16>)
        outs(%dest : tensor<1024x512xf32>) {match_status = "unmatched"} -> tensor<1024x512xf32>
  return %res : tensor<1024x512xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_contraction_and_size(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %m, %n, %k = transform.iree.match.contraction %op,
      lhs_type = f16, rhs_type = f16, output_type = f32 : !transform.any_op -> !transform.param<i64>

    transform.iree.match.dims_equal %m, [4096] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [2048] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [8192] : !transform.param<i64>

    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "match_status" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_contraction_and_size -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// Verify MatchSizeEqualsOp with multiple batch dimensions.

#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @op_contract_multi_dims
func.func @op_contract_multi_dims(%input0: tensor<2x4x32x64xf16>, %input1: tensor<2x4x32x64xf16>, %dest: tensor<2x4x32x32xf32>) -> tensor<2x4x32x32xf32> {
  // CHECK-NEXT: linalg.contract
  // CHECK-SAME:   match_status = "matched"
  %res = linalg.contract
        indexing_maps = [#map0, #map1, #map2]
        ins(%input0, %input1 : tensor<2x4x32x64xf16>, tensor<2x4x32x64xf16>)
        outs(%dest : tensor<2x4x32x32xf32>) {match_status = "unmatched"} -> tensor<2x4x32x32xf32>
  return %res : tensor<2x4x32x32xf32>
}

// CHECK-LABEL: func.func @op_contract_mismatch_dims
func.func @op_contract_mismatch_dims(%input0: tensor<3x5x16x128xf16>, %input1: tensor<3x5x16x128xf16>, %dest: tensor<3x5x16x16xf32>) -> tensor<3x5x16x16xf32> {
  // CHECK-NEXT: linalg.contract
  // CHECK-SAME:   match_status = "unmatched"
  %res = linalg.contract
        indexing_maps = [#map0, #map1, #map2]
        ins(%input0, %input1 : tensor<3x5x16x128xf16>, tensor<3x5x16x128xf16>)
        outs(%dest : tensor<3x5x16x16xf32>) {match_status = "unmatched"} -> tensor<3x5x16x16xf32>
  return %res : tensor<3x5x16x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_multi_dimensions(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch_dims, %m_dims, %n_dims, %k_dims = transform.iree.match.contraction %op,
      lhs_type = f16, rhs_type = f16, output_type = f32 : !transform.any_op -> !transform.param<i64>
    // Test multiple dimensions.
    // %batch_dims = [2, 4] - check if first batch dim is 2 and second is 4.
    transform.iree.match.dims_equal %batch_dims, [2, 4] : !transform.param<i64>
    transform.iree.match.dims_equal %m_dims, [32] : !transform.param<i64>
    transform.iree.match.dims_equal %n_dims, [32] : !transform.param<i64>
    transform.iree.match.dims_equal %k_dims, [64] : !transform.param<i64>

    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "match_status" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_multi_dimensions -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// Verify MatchSizeEqualsOp fails when array lengths don't match.

// CHECK-LABEL: func.func @op_matmul_size_mismatch
func.func @op_matmul_size_mismatch(%input0: tensor<512x1024xf16>, %input1: tensor<1024x256xf16>, %dest: tensor<512x256xf32>) -> tensor<512x256xf32> {
  // CHECK-NEXT: linalg.matmul
  // CHECK-SAME:   match_status = "unmatched"
  %res = linalg.matmul
        ins(%input0, %input1 : tensor<512x1024xf16>, tensor<1024x256xf16>)
        outs(%dest : tensor<512x256xf32>) {match_status = "unmatched"} -> tensor<512x256xf32>
  return %res : tensor<512x256xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_size_mismatch(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %m, %n, %k = transform.iree.match.contraction %op,
      lhs_type = f16, rhs_type = f16, output_type = f32 : !transform.any_op -> !transform.param<i64>
    // This should fail because %m contains [512] but we expect [512, 256] (different array lengths).
    transform.iree.match.dims_equal %m, [512, 256] : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "match_status" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_size_mismatch -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: func.func @broadcast_mmt_cases
func.func @broadcast_mmt_cases(
  %lhs_r: tensor<4x8x512xi8>,   %rhs_r: tensor<1024x512xi8>, %out_r: tensor<4x8x1024xi32>,
  %lhs_l: tensor<1024x512xi8>,  %rhs_l: tensor<4x8x512xi8>,  %out_l: tensor<4x8x1024xi32>
) -> (tensor<4x8x1024xi32>, tensor<4x8x1024xi32>) {
  // CHECK: linalg.generic
  // CHECK-SAME: match_status = "matched"
  %rhs_res = linalg.generic
    { indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel","parallel","parallel","reduction"],
      match_status = "unmatched" }
    ins(%lhs_r, %rhs_r : tensor<4x8x512xi8>, tensor<1024x512xi8>)
    outs(%out_r : tensor<4x8x1024xi32>) {
  ^bb0(%in_l: i8, %in_r: i8, %acc: i32):
    %l = arith.extsi %in_l : i8 to i32
    %r = arith.extsi %in_r : i8 to i32
    %m = arith.muli %l, %r : i32
    %a = arith.addi %acc, %m : i32
    linalg.yield %a : i32
  } -> tensor<4x8x1024xi32>

  // CHECK: linalg.generic
  // CHECK-SAME: match_status = "matched"
  %lhs_res = linalg.generic
    { indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ],
      iterator_types = ["parallel","parallel","parallel","reduction"],
      match_status = "unmatched" }
    ins(%lhs_l, %rhs_l : tensor<1024x512xi8>, tensor<4x8x512xi8>)
    outs(%out_l : tensor<4x8x1024xi32>) {
  ^bb0(%in_l2: i8, %in_r2: i8, %acc2: i32):
    %l2 = arith.extsi %in_l2 : i8 to i32
    %r2 = arith.extsi %in_r2 : i8 to i32
    %m2 = arith.muli %l2, %r2 : i32
    %a2 = arith.addi %acc2, %m2 : i32
    linalg.yield %a2 : i32
  } -> tensor<4x8x1024xi32>

  return %rhs_res, %lhs_res : tensor<4x8x1024xi32>, tensor<4x8x1024xi32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_broadcast_rhs(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %m, %n, %k = transform.iree.match.contraction %op,
      lhs_type = i8, rhs_type = i8, output_type = i32,
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [-1, 8] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [1024] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [512] : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @match_broadcast_lhs(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %m, %n, %k = transform.iree.match.contraction %op,
      lhs_type = i8, rhs_type = i8, output_type = i32,
      indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
      ] : !transform.any_op -> !transform.param<i64>
    transform.iree.match.dims_equal %batch, [] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [1024] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [-1, 8] : !transform.param<i64>
    transform.iree.match.dims_equal %k, [512] : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %s = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "match_status" = %s : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_broadcast_rhs -> @annotate,
        @match_broadcast_lhs -> @annotate
      : (!transform.any_op) -> !transform.any_op
    transform.yield
 }
}

// -----

// Verify that the basic convolution matcher works.

#map_nhwc_hwcf_input = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map_nhwc_hwcf_filter = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map_nhwc_hwcf_output = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#map_nchw_fchw_input = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#map_nchw_fchw_filter = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map_nchw_fchw_output = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @conv_ops
func.func @conv_ops(
    %input_nhwc: tensor<2x34x34x64xf32>,
    %filter_nhwc: tensor<3x3x64x128xf32>,
    %output_nhwc: tensor<2x32x32x128xf32>,
    %input_nchw: tensor<4x32x112x112xf16>,
    %filter_nchw: tensor<64x32x7x7xf16>,
    %output_nchw: tensor<4x64x106x106xf16>,
    %input_mm: tensor<32x64xf32>,
    %filter_mm: tensor<64x32xf32>,
    %output_mm: tensor<32x32xf32>) -> (tensor<2x32x32x128xf32>, tensor<4x64x106x106xf16>, tensor<32x32xf32>) {

  // CHECK: linalg.conv_2d_nhwc_hwcf
  // CHECK-SAME:   match_status = "matched"
  %res1 = linalg.conv_2d_nhwc_hwcf
        ins(%input_nhwc, %filter_nhwc : tensor<2x34x34x64xf32>, tensor<3x3x64x128xf32>)
        outs(%output_nhwc : tensor<2x32x32x128xf32>) {match_status = "unmatched"} -> tensor<2x32x32x128xf32>

  // CHECK: linalg.conv_2d_nchw_fchw
  // CHECK-SAME:   match_status = "matched"
  %res2 = linalg.conv_2d_nchw_fchw
        ins(%input_nchw, %filter_nchw : tensor<4x32x112x112xf16>, tensor<64x32x7x7xf16>)
        outs(%output_nchw : tensor<4x64x106x106xf16>) {match_status = "unmatched"} -> tensor<4x64x106x106xf16>

  // CHECK: linalg.matmul
  // CHECK-SAME:   match_status = "unmatched"
  %res3 = linalg.matmul
        ins(%input_mm, %filter_mm : tensor<32x64xf32>, tensor<64x32xf32>)
        outs(%output_mm : tensor<32x32xf32>) {match_status = "unmatched"} -> tensor<32x32xf32>

  return %res1, %res2, %res3 : tensor<2x32x32x128xf32>, tensor<4x64x106x106xf16>, tensor<32x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_conv_nhwc_hwcf(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %out_img, %out_ch, %filt, %in_ch, %depth, %strides, %dilations =
      transform.iree.match.convolution %op,
        lhs_type = f32, rhs_type = f32, output_type = f32,
        indexing_maps = [#map_nhwc_hwcf_input, #map_nhwc_hwcf_filter, #map_nhwc_hwcf_output] :
        !transform.any_op -> !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @match_conv_nchw_fchw(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %out_img, %out_ch, %filt, %in_ch, %depth, %strides, %dilations =
      transform.iree.match.convolution %op,
        lhs_type = f16, rhs_type = f16, output_type = f16,
        indexing_maps = [#map_nchw_fchw_input, #map_nchw_fchw_filter, #map_nchw_fchw_output] :
        !transform.any_op -> !transform.param<i64>
    %c4 = transform.param.constant 4 : i64 -> !transform.param<i64>
    transform.match.param.cmpi eq %batch, %c4 : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "match_status" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_conv_nhwc_hwcf -> @annotate,
        @match_conv_nchw_fchw -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// Verify dimension size constraints for the convolution op with lowering config.

#map_nhwc_hwcf_input = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map_nhwc_hwcf_filter = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>
#map_nhwc_hwcf_output = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

#lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1]}>

// CHECK-LABEL: func.func @conv_constraints
func.func @conv_constraints(
    %in1: tensor<1x224x224x3xf16>,
    %filt1: tensor<7x7x3x64xf16>,
    %out1: tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32> {

  // CHECK: linalg.conv_2d_nhwc_hwcf
  // CHECK-SAME:   lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1]}>
  // CHECK-SAME:   match_status = "matched"
  %res = linalg.conv_2d_nhwc_hwcf {lowering_config = #lowering_config}
        ins(%in1, %filt1 : tensor<1x224x224x3xf16>, tensor<7x7x3x64xf16>)
        outs(%out1 : tensor<1x112x112x64xf32>) {match_status = "unmatched"} -> tensor<1x112x112x64xf32>

  return %res : tensor<1x112x112x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_conv_all_dims(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %out_img, %out_ch, %filt, %in_ch, %depth, %strides, %dilations =
      transform.iree.match.convolution %op,
        lhs_type = f16, rhs_type = f16, output_type = f32,
        indexing_maps = [#map_nhwc_hwcf_input, #map_nhwc_hwcf_filter, #map_nhwc_hwcf_output] :
        !transform.any_op -> !transform.param<i64>

    transform.iree.match.dims_equal %batch, [1] : !transform.param<i64>
    transform.iree.match.dims_equal %out_img, [112, 112] : !transform.param<i64>
    transform.iree.match.dims_equal %out_ch, [64] : !transform.param<i64>
    transform.iree.match.dims_equal %filt, [7, 7] : !transform.param<i64>
    transform.iree.match.dims_equal %in_ch, [3] : !transform.param<i64>
    transform.iree.match.dims_equal %depth, [] : !transform.param<i64>
    transform.iree.match.dims_equal %strides, [1, 1] : !transform.param<i64>
    transform.iree.match.dims_equal %dilations, [1, 1] : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "match_status" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_conv_all_dims -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// Verify indexing maps mismatching for the convolution op.

#map_nhwc_hwcf_input = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>
#map_nhwc_hwcf_output = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

// Wrong filter map with transposed channels: (d4, d5, d3, d6) instead of (d4, d5, d6, d3).
#map_wrong_filter = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d3, d6)>

// CHECK-LABEL: func.func @indexing_maps_test
func.func @indexing_maps_test(
    %input: tensor<1x224x224x3xf32>,
    %filter: tensor<7x7x3x64xf32>,
    %output: tensor<1x218x218x64xf32>) -> tensor<1x218x218x64xf32> {

  // CHECK: linalg.conv_2d_nhwc_hwcf
  // CHECK-SAME:   maps_match = "unmatched"
  %res = linalg.conv_2d_nhwc_hwcf
        ins(%input, %filter : tensor<1x224x224x3xf32>, tensor<7x7x3x64xf32>)
        outs(%output : tensor<1x218x218x64xf32>) {maps_match = "unmatched"} -> tensor<1x218x218x64xf32>

  return %res : tensor<1x218x218x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_with_wrong_maps(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %out_img, %out_ch, %filt, %in_ch, %depth, %strides, %dilations =
      transform.iree.match.convolution %op,
        lhs_type = f32, rhs_type = f32, output_type = f32,
        indexing_maps = [#map_nhwc_hwcf_input, #map_wrong_filter, #map_nhwc_hwcf_output] :
        !transform.any_op -> !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "maps_match" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_with_wrong_maps -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// Verify that the basic attention matcher works.

#map_query = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
#map_key = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
#map_value = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
#map_output = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @attention_ops
func.func @attention_ops(
    %query: tensor<2x10x6x4xf16>,
    %key: tensor<2x10x4x4xf16>,
    %value: tensor<2x10x4x4xf16>,
    %output: tensor<2x10x6x4xf16>,
    %input_mm: tensor<32x64xf32>,
    %filter_mm: tensor<64x32xf32>,
    %output_mm: tensor<32x32xf32>) -> (tensor<2x10x6x4xf16>, tensor<32x32xf32>) {

  // CHECK: iree_linalg_ext.attention
  // CHECK-SAME:   match_status = "matched"
  %res1 = iree_linalg_ext.attention {indexing_maps = [#map_query, #map_key, #map_value, #map_output], match_status = "unmatched"}
      ins(%query, %key, %value : tensor<2x10x6x4xf16>, tensor<2x10x4x4xf16>, tensor<2x10x4x4xf16>)
      outs(%output : tensor<2x10x6x4xf16>) {
    ^bb0(%arg: f32):
      iree_linalg_ext.yield %arg : f32
    } -> tensor<2x10x6x4xf16>

  // CHECK: linalg.matmul
  // CHECK-SAME:   match_status = "unmatched"
  %res2 = linalg.matmul
        ins(%input_mm, %filter_mm : tensor<32x64xf32>, tensor<64x32xf32>)
        outs(%output_mm : tensor<32x32xf32>) {match_status = "unmatched"} -> tensor<32x32xf32>

  return %res1, %res2 : tensor<2x10x6x4xf16>, tensor<32x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_attention(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %m, %n, %k1, %k2 =
      transform.iree.match.attention %op,
        query_type = f16, key_type = f16, value_type = f16, output_type = f16,
        indexing_maps = [#map_query, #map_key, #map_value, #map_output] :
        !transform.any_op -> !transform.param<i64>

    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "match_status" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_attention -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// Verify dimension size constraints for the attention op with lowering config.

#map_query = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
#map_key = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
#map_value = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
#map_output = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

#lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1]}>

// CHECK-LABEL: func.func @attention_constraints
func.func @attention_constraints(
    %query: tensor<2x10x6x4xf16>,
    %key: tensor<2x10x4x4xf16>,
    %value: tensor<2x10x4x4xf16>,
    %output: tensor<2x10x6x4xf16>) -> tensor<2x10x6x4xf16> {

  // CHECK: iree_linalg_ext.attention
  // CHECK-SAME:   lowering_config = #iree_gpu.lowering_config<{promote_operands = [0, 1]}>
  // CHECK-SAME:   match_status = "matched"
  %res = iree_linalg_ext.attention {
      indexing_maps = [#map_query, #map_key, #map_value, #map_output],
      lowering_config = #lowering_config,
      match_status = "unmatched"}
      ins(%query, %key, %value : tensor<2x10x6x4xf16>, tensor<2x10x4x4xf16>, tensor<2x10x4x4xf16>)
      outs(%output : tensor<2x10x6x4xf16>) {
    ^bb0(%arg: f32):
      iree_linalg_ext.yield %arg : f32
    } -> tensor<2x10x6x4xf16>

  return %res : tensor<2x10x6x4xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_attention_all_dims(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %m, %n, %k1, %k2 =
      transform.iree.match.attention %op,
        query_type = f16, key_type = f16, value_type = f16, output_type = f16,
        indexing_maps = [#map_query, #map_key, #map_value, #map_output] :
        !transform.any_op -> !transform.param<i64>

    transform.iree.match.dims_equal %batch, [2, 10] : !transform.param<i64>
    transform.iree.match.dims_equal %m, [6] : !transform.param<i64>
    transform.iree.match.dims_equal %k1, [4] : !transform.param<i64>
    transform.iree.match.dims_equal %k2, [4] : !transform.param<i64>
    transform.iree.match.dims_equal %n, [4] : !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "match_status" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_attention_all_dims -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

// Verify indexing maps mismatching for the attention op.

#map_query = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>
#map_key = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>
#map_value = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>
#map_output = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>

#map_wrong_key = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d5)>

// CHECK-LABEL: func.func @indexing_maps_test
func.func @indexing_maps_test(
    %query: tensor<2x10x6x4xf16>,
    %key: tensor<2x10x4x4xf16>,
    %value: tensor<2x10x4x4xf16>,
    %output: tensor<2x10x6x4xf16>) -> tensor<2x10x6x4xf16> {

  // CHECK: iree_linalg_ext.attention
  // CHECK-SAME:   maps_match = "unmatched"
  %res = iree_linalg_ext.attention {indexing_maps = [#map_query, #map_key, #map_value, #map_output], maps_match = "unmatched"}
      ins(%query, %key, %value : tensor<2x10x6x4xf16>, tensor<2x10x4x4xf16>, tensor<2x10x4x4xf16>)
      outs(%output : tensor<2x10x6x4xf16>) {
    ^bb0(%arg: f32):
      iree_linalg_ext.yield %arg : f32
    } -> tensor<2x10x6x4xf16>

  return %res : tensor<2x10x6x4xf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @match_with_wrong_maps(%op: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %batch, %m, %n, %k1, %k2 =
      transform.iree.match.attention %op,
        query_type = f16, key_type = f16, value_type = f16, output_type = f16,
        indexing_maps = [#map_query, #map_wrong_key, #map_value, #map_output] :
        !transform.any_op -> !transform.param<i64>
    transform.yield %op : !transform.any_op
  }

  transform.named_sequence @annotate(%op: !transform.any_op {transform.readonly}) {
    %0 = transform.param.constant "matched" -> !transform.any_param
    transform.annotate %op "maps_match" = %0 : !transform.any_op, !transform.any_param
    transform.yield
  }

  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    transform.foreach_match in %module
        @match_with_wrong_maps -> @annotate
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
