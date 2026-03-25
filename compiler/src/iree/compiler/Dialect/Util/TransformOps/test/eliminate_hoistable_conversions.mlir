// RUN: iree-opt --transform-interpreter --split-input-file %s | FileCheck %s

// CHECK-LABEL: @cancel_inverse_pair
// CHECK-SAME: %[[ARG:.+]]: vector<4xf32>
// CHECK-NEXT: return %[[ARG]] : vector<4xf32>
module attributes { transform.with_named_sequence } {
  func.func @cancel_inverse_pair(%arg0 : vector<4xf32>) -> vector<4xf32> {
    %0 = util.hoistable_conversion "to" inverts("from")
        (%a: vector<4xf32> = %arg0) : (vector<4xf32>) -> vector<2x2xf32> {
      %1 = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
      util.return %1 : vector<2x2xf32>
    }
    %1 = util.hoistable_conversion "from" inverts("to")
        (%b: vector<2x2xf32> = %0) : (vector<2x2xf32>) -> vector<4xf32> {
      %2 = vector.shape_cast %b : vector<2x2xf32> to vector<4xf32>
      util.return %2 : vector<4xf32>
    }
    return %1 : vector<4xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.util.eliminate_hoistable_conversions %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @hoist_from_loop
// CHECK-SAME: %[[INIT:.+]]: vector<4xf32>
// CHECK: %[[SC0:.+]] = vector.shape_cast %[[INIT]] : vector<4xf32> to vector<2x2xf32>
// CHECK: %[[LOOP:.+]]:2 = scf.for {{.+}} iter_args(%{{.+}} = %[[INIT]], %[[ITER:.+]] = %[[SC0]])
// CHECK:   %[[ADD:.+]] = arith.addf %[[ITER]], %[[ITER]]
// CHECK:   scf.yield {{.+}}, %[[ADD]]
// CHECK: %[[SC1:.+]] = vector.shape_cast %[[LOOP]]#1 : vector<2x2xf32> to vector<4xf32>
// CHECK: return %[[SC1]]
module attributes { transform.with_named_sequence } {
  func.func @hoist_from_loop(%init : vector<4xf32>) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %init) -> vector<4xf32> {
      %0 = util.hoistable_conversion "to" inverts("from")
          (%a: vector<4xf32> = %acc) : (vector<4xf32>) -> vector<2x2xf32> {
        %sc = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
        util.return %sc : vector<2x2xf32>
      }
      %1 = arith.addf %0, %0 : vector<2x2xf32>
      %2 = util.hoistable_conversion "from" inverts("to")
          (%b: vector<2x2xf32> = %1) : (vector<2x2xf32>) -> vector<4xf32> {
        %sc = vector.shape_cast %b : vector<2x2xf32> to vector<4xf32>
        util.return %sc : vector<4xf32>
      }
      scf.yield %2 : vector<4xf32>
    }
    return %result : vector<4xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.util.eliminate_hoistable_conversions %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @standalone_inline
// CHECK-SAME: %[[ARG:.+]]: vector<4xf32>
// CHECK-NEXT: %[[SC:.+]] = vector.shape_cast %[[ARG]] : vector<4xf32> to vector<2x2xf32>
// CHECK-NEXT: return %[[SC]]
module attributes { transform.with_named_sequence } {
  func.func @standalone_inline(%arg0 : vector<4xf32>) -> vector<2x2xf32> {
    %0 = util.hoistable_conversion "to" inverts("from")
        (%a: vector<4xf32> = %arg0) : (vector<4xf32>) -> vector<2x2xf32> {
      %1 = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
      util.return %1 : vector<2x2xf32>
    }
    return %0 : vector<2x2xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.util.eliminate_hoistable_conversions %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @chained_cancellation
// CHECK-SAME: %[[ARG:.+]]: vector<8xf16>
// CHECK-NEXT: return %[[ARG]]
module attributes { transform.with_named_sequence } {
  func.func @chained_cancellation(%arg0 : vector<8xf16>) -> vector<8xf16> {
    %0 = util.hoistable_conversion "sc_to" inverts("sc_from")
        (%a: vector<8xf16> = %arg0) : (vector<8xf16>) -> vector<4x2xf16> {
      %sc = vector.shape_cast %a : vector<8xf16> to vector<4x2xf16>
      util.return %sc : vector<4x2xf16>
    }
    %1 = util.hoistable_conversion "interleave" inverts("deinterleave")
        (%b: vector<4x2xf16> = %0) : (vector<4x2xf16>) -> vector<4x2xf16> {
      util.return %b : vector<4x2xf16>
    }
    %2 = util.hoistable_conversion "deinterleave" inverts("interleave")
        (%c: vector<4x2xf16> = %1) : (vector<4x2xf16>) -> vector<4x2xf16> {
      util.return %c : vector<4x2xf16>
    }
    %3 = util.hoistable_conversion "sc_from" inverts("sc_to")
        (%d: vector<4x2xf16> = %2) : (vector<4x2xf16>) -> vector<8xf16> {
      %sc = vector.shape_cast %d : vector<4x2xf16> to vector<8xf16>
      util.return %sc : vector<8xf16>
    }
    return %3 : vector<8xf16>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.util.eliminate_hoistable_conversions %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Only one of two iter_args has F/G wrapping.
// CHECK-LABEL: @partial_hoist
// CHECK-SAME: %[[INIT_A:.+]]: vector<4xf32>, %[[INIT_B:.+]]: f32
// CHECK: %[[SC0:.+]] = vector.shape_cast %[[INIT_A]] : vector<4xf32> to vector<2x2xf32>
// CHECK: scf.for {{.+}} iter_args({{.+}} = %[[INIT_A]], %[[B_ARG:.+]] = %[[INIT_B]], {{.+}} = %[[SC0]])
// CHECK:   arith.addf
// CHECK:   %[[B_NEW:.+]] = arith.addf %[[B_ARG]]
// CHECK:   scf.yield {{.+}}, %[[B_NEW]],
// CHECK: vector.shape_cast %{{.+}} : vector<2x2xf32> to vector<4xf32>
module attributes { transform.with_named_sequence } {
  func.func @partial_hoist(%init_a : vector<4xf32>, %init_b : f32) -> (vector<4xf32>, f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %result:2 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc_a = %init_a, %acc_b = %init_b) -> (vector<4xf32>, f32) {
      %0 = util.hoistable_conversion "to" inverts("from")
          (%a: vector<4xf32> = %acc_a) : (vector<4xf32>) -> vector<2x2xf32> {
        %sc = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
        util.return %sc : vector<2x2xf32>
      }
      %1 = arith.addf %0, %0 : vector<2x2xf32>
      %2 = util.hoistable_conversion "from" inverts("to")
          (%b: vector<2x2xf32> = %1) : (vector<2x2xf32>) -> vector<4xf32> {
        %sc = vector.shape_cast %b : vector<2x2xf32> to vector<4xf32>
        util.return %sc : vector<4xf32>
      }
      %3 = arith.addf %acc_b, %acc_b : f32
      scf.yield %2, %3 : vector<4xf32>, f32
    }
    return %result#0, %result#1 : vector<4xf32>, f32
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.util.eliminate_hoistable_conversions %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// G before a loop cancels with the hoisted F.
// CHECK-LABEL: @compose_g_before_loop
// CHECK-SAME: %[[INIT:.+]]: vector<2x2xf32>
// CHECK-NOT: util.hoistable_conversion
// CHECK: scf.for
// CHECK:   arith.addf
// CHECK:   scf.yield
// CHECK: vector.shape_cast
// CHECK: return
module attributes { transform.with_named_sequence } {
  func.func @compose_g_before_loop(%init : vector<2x2xf32>) -> vector<4xf32> {
    %pre = util.hoistable_conversion "from" inverts("to")
        (%x: vector<2x2xf32> = %init) : (vector<2x2xf32>) -> vector<4xf32> {
      %sc = vector.shape_cast %x : vector<2x2xf32> to vector<4xf32>
      util.return %sc : vector<4xf32>
    }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %pre) -> vector<4xf32> {
      %0 = util.hoistable_conversion "to" inverts("from")
          (%a: vector<4xf32> = %acc) : (vector<4xf32>) -> vector<2x2xf32> {
        %sc = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
        util.return %sc : vector<2x2xf32>
      }
      %1 = arith.addf %0, %0 : vector<2x2xf32>
      %2 = util.hoistable_conversion "from" inverts("to")
          (%b: vector<2x2xf32> = %1) : (vector<2x2xf32>) -> vector<4xf32> {
        %sc = vector.shape_cast %b : vector<2x2xf32> to vector<4xf32>
        util.return %sc : vector<4xf32>
      }
      scf.yield %2 : vector<4xf32>
    }
    return %result : vector<4xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.util.eliminate_hoistable_conversions %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// F after a loop cancels with the hoisted G.
// CHECK-LABEL: @compose_f_after_loop
// CHECK-SAME: %[[INIT:.+]]: vector<4xf32>
// CHECK: vector.shape_cast
// CHECK: scf.for
// CHECK:   arith.addf
// CHECK:   scf.yield
// CHECK-NOT: util.hoistable_conversion
// CHECK: return
module attributes { transform.with_named_sequence } {
  func.func @compose_f_after_loop(%init : vector<4xf32>) -> vector<2x2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %init) -> vector<4xf32> {
      %0 = util.hoistable_conversion "to" inverts("from")
          (%a: vector<4xf32> = %acc) : (vector<4xf32>) -> vector<2x2xf32> {
        %sc = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
        util.return %sc : vector<2x2xf32>
      }
      %1 = arith.addf %0, %0 : vector<2x2xf32>
      %2 = util.hoistable_conversion "from" inverts("to")
          (%b: vector<2x2xf32> = %1) : (vector<2x2xf32>) -> vector<4xf32> {
        %sc = vector.shape_cast %b : vector<2x2xf32> to vector<4xf32>
        util.return %sc : vector<4xf32>
      }
      scf.yield %2 : vector<4xf32>
    }
    %post = util.hoistable_conversion "to" inverts("from")
        (%x: vector<4xf32> = %result) : (vector<4xf32>) -> vector<2x2xf32> {
      %sc = vector.shape_cast %x : vector<4xf32> to vector<2x2xf32>
      util.return %sc : vector<2x2xf32>
    }
    return %post : vector<2x2xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.util.eliminate_hoistable_conversions %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
