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
