// RUN: iree-opt --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @cancel_inverse_pair
// CHECK-SAME: %[[ARG:.+]]: vector<4xf32>
// CHECK-NEXT: return %[[ARG]] : vector<4xf32>
module attributes { transform.with_named_sequence } {
  func.func @cancel_inverse_pair(%arg0 : vector<4xf32>) -> vector<4xf32> {
    %0 = util.hoistable_conversion "to" inverts("from")
        (%a = %arg0) : (vector<4xf32>) -> vector<2x2xf32> {
      %1 = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
      util.return %1 : vector<2x2xf32>
    }
    %1 = util.hoistable_conversion "from" inverts("to")
        (%b = %0) : (vector<2x2xf32>) -> vector<4xf32> {
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
// CHECK-DAG: %[[SC0:.+]] = vector.shape_cast %[[INIT]] : vector<4xf32> to vector<2x2xf32>
// CHECK-DAG: %[[POISON:.+]] = ub.poison : vector<4xf32>
// CHECK: %[[LOOP:.+]]:2 = scf.for {{.+}} iter_args(%{{.+}} = %[[POISON]], %[[ITER:.+]] = %[[SC0]])
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
          (%a = %acc) : (vector<4xf32>) -> vector<2x2xf32> {
        %sc = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
        util.return %sc : vector<2x2xf32>
      }
      %1 = arith.addf %0, %0 : vector<2x2xf32>
      %2 = util.hoistable_conversion "from" inverts("to")
          (%b = %1) : (vector<2x2xf32>) -> vector<4xf32> {
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
        (%a = %arg0) : (vector<4xf32>) -> vector<2x2xf32> {
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
        (%a = %arg0) : (vector<8xf16>) -> vector<4x2xf16> {
      %sc = vector.shape_cast %a : vector<8xf16> to vector<4x2xf16>
      util.return %sc : vector<4x2xf16>
    }
    %1 = util.hoistable_conversion "interleave" inverts("deinterleave")
        (%b = %0) : (vector<4x2xf16>) -> vector<4x2xf16> {
      util.return %b : vector<4x2xf16>
    }
    %2 = util.hoistable_conversion "deinterleave" inverts("interleave")
        (%c = %1) : (vector<4x2xf16>) -> vector<4x2xf16> {
      util.return %c : vector<4x2xf16>
    }
    %3 = util.hoistable_conversion "sc_from" inverts("sc_to")
        (%d = %2) : (vector<4x2xf16>) -> vector<8xf16> {
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
// CHECK-DAG: %[[SC0:.+]] = vector.shape_cast %[[INIT_A]] : vector<4xf32> to vector<2x2xf32>
// CHECK-DAG: %[[POISON:.+]] = ub.poison : vector<4xf32>
// CHECK: scf.for {{.+}} iter_args({{.+}} = %[[POISON]], %[[B_ARG:.+]] = %[[INIT_B]], {{.+}} = %[[SC0]])
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
          (%a = %acc_a) : (vector<4xf32>) -> vector<2x2xf32> {
        %sc = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
        util.return %sc : vector<2x2xf32>
      }
      %1 = arith.addf %0, %0 : vector<2x2xf32>
      %2 = util.hoistable_conversion "from" inverts("to")
          (%b = %1) : (vector<2x2xf32>) -> vector<4xf32> {
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
        (%x = %init) : (vector<2x2xf32>) -> vector<4xf32> {
      %sc = vector.shape_cast %x : vector<2x2xf32> to vector<4xf32>
      util.return %sc : vector<4xf32>
    }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %pre) -> vector<4xf32> {
      %0 = util.hoistable_conversion "to" inverts("from")
          (%a = %acc) : (vector<4xf32>) -> vector<2x2xf32> {
        %sc = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
        util.return %sc : vector<2x2xf32>
      }
      %1 = arith.addf %0, %0 : vector<2x2xf32>
      %2 = util.hoistable_conversion "from" inverts("to")
          (%b = %1) : (vector<2x2xf32>) -> vector<4xf32> {
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
          (%a = %acc) : (vector<4xf32>) -> vector<2x2xf32> {
        %sc = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
        util.return %sc : vector<2x2xf32>
      }
      %1 = arith.addf %0, %0 : vector<2x2xf32>
      %2 = util.hoistable_conversion "from" inverts("to")
          (%b = %1) : (vector<2x2xf32>) -> vector<4xf32> {
        %sc = vector.shape_cast %b : vector<2x2xf32> to vector<4xf32>
        util.return %sc : vector<4xf32>
      }
      scf.yield %2 : vector<4xf32>
    }
    %post = util.hoistable_conversion "to" inverts("from")
        (%x = %result) : (vector<4xf32>) -> vector<2x2xf32> {
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

// -----

// Conversions from multiple loops cancel in the middle
// CHECK-LABEL: @fold_communication_between_two_loops
// CHECK-SAME: %[[INIT:.+]]: vector<2x2xf32>
// CHECK: %[[SC0:.+]] = vector.shape_cast %[[INIT]]
// CHECK: %[[RESULT1:.+]]:2 = scf.for {{.*}} %[[SC0]]
// CHECK-NOT: vector.shape_castw
// CHECK:   arith.addf
// CHECK-NOT:   vector.shape_castw
// CHECK:   scf.yield
// CHECK: %[[RESULT2:.+]]:2 = scf.for {{.*}} %[[RESULT1]]#1
// CHECK-NOT:   vector.shape_castw
// CHECK:   arith.addf
// CHECK-NOT:   vector.shape_castw
// CHECK:   scf.yield
// CHECK-NOT: util.hoistable_conversion
// CHECK: %[[RET:.+]] = vector.shape_cast %[[RESULT2]]#1
// CHECK: return %[[RET]]
module attributes { transform.with_named_sequence } {
  func.func @fold_communication_between_two_loops(%init : vector<2x2xf32>) -> vector<2x2xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %result1 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %init) -> vector<2x2xf32> {
      %0 = util.hoistable_conversion "to" inverts("from")
          (%a = %acc) : (vector<2x2xf32>) -> vector<4xf32> {
        %sc = vector.shape_cast %a : vector<2x2xf32> to vector<4xf32>
        util.return %sc : vector<4xf32>
      }
      %1 = arith.addf %0, %0 : vector<4xf32>
      %2 = util.hoistable_conversion "from" inverts("to")
          (%b = %1) : (vector<4xf32>) -> vector<2x2xf32> {
        %sc = vector.shape_cast %b : vector<4xf32> to vector<2x2xf32>
        util.return %sc : vector<2x2xf32>
      }
      scf.yield %2 : vector<2x2xf32>
    }
    %result2 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %result1) -> vector<2x2xf32> {
      %0 = util.hoistable_conversion "to" inverts("from")
          (%a = %acc) : (vector<2x2xf32>) -> vector<4xf32> {
        %sc = vector.shape_cast %a : vector<2x2xf32> to vector<4xf32>
        util.return %sc : vector<4xf32>
      }
      %1 = arith.addf %0, %0 : vector<4xf32>
      %2 = util.hoistable_conversion "from" inverts("to")
          (%b = %1) : (vector<4xf32>) -> vector<2x2xf32> {
        %sc = vector.shape_cast %b : vector<4xf32> to vector<2x2xf32>
        util.return %sc : vector<2x2xf32>
      }
      scf.yield %2 : vector<2x2xf32>
    }

    return %result2 : vector<2x2xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.util.eliminate_hoistable_conversions %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @nested_loops
// CHECK-SAME: %[[INIT:.+]]: vector<4xf32>
// CHECK-DAG: %[[SC0:.+]] = vector.shape_cast %[[INIT]] : vector<4xf32> to vector<2x2xf32>
// CHECK-DAG: %[[POISON:.+]] = ub.poison : vector<4xf32>
// CHECK: %[[OUTER:.+]]:2 = scf.for {{.+}} iter_args({{.+}} = %[[POISON]], %[[OITER:.+]] = %[[SC0]])
// CHECK:   scf.for {{.+}} iter_args({{.+}} = %[[OITER]]
// CHECK:     arith.addf
// CHECK:     scf.yield
// CHECK:   scf.yield
// CHECK: vector.shape_cast %[[OUTER]]#1 : vector<2x2xf32> to vector<4xf32>
// CHECK-NOT: util.hoistable_conversion
module attributes { transform.with_named_sequence } {
  func.func @nested_loops(%init : vector<4xf32>) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c10 = arith.constant 10 : index
    %outer = scf.for %oi = %c0 to %c10 step %c1 iter_args(%oacc = %init) -> vector<4xf32> {
      %inner = scf.for %ii = %c0 to %c5 step %c1 iter_args(%iacc = %oacc) -> vector<4xf32> {
        %0 = util.hoistable_conversion "to" inverts("from")
            (%a = %iacc) : (vector<4xf32>) -> vector<2x2xf32> {
          %sc = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
          util.return %sc : vector<2x2xf32>
        }
        %1 = arith.addf %0, %0 : vector<2x2xf32>
        %2 = util.hoistable_conversion "from" inverts("to")
            (%b = %1) : (vector<2x2xf32>) -> vector<4xf32> {
          %sc = vector.shape_cast %b : vector<2x2xf32> to vector<4xf32>
          util.return %sc : vector<4xf32>
        }
        scf.yield %2 : vector<4xf32>
      }
      scf.yield %inner : vector<4xf32>
    }
    return %outer : vector<4xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.util.eliminate_hoistable_conversions %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: @unhoisted_remark
// CHECK: vector.shape_cast
// CHECK-NOT: util.hoistable_conversion
module attributes { transform.with_named_sequence } {
  func.func @unhoisted_remark(%init : vector<4xf32>) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%acc = %init) -> vector<4xf32> {
      // expected-remark @+1 {{hoistable_conversion was not hoisted or cancelled; inlining in place}}
      %0 = util.hoistable_conversion "orphan_tag" inverts("no_matching_inverse")
          (%a = %acc) : (vector<4xf32>) -> vector<2x2xf32> {
        %sc = vector.shape_cast %a : vector<4xf32> to vector<2x2xf32>
        util.return %sc : vector<2x2xf32>
      }
      %flat = vector.shape_cast %0 : vector<2x2xf32> to vector<4xf32>
      scf.yield %flat : vector<4xf32>
    }
    return %result : vector<4xf32>
  }
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.util.eliminate_hoistable_conversions %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
