// RUN: iree-dialects-opt -split-input-file -convert-iree-pydm-to-iree %s | FileCheck  --dump-input-filter=all %s

// CHECK-LABEL: @bool_to_pred
// NOTE: Also tests cond_br conversion.
iree_pydm.func @bool_to_pred(%arg0 : !iree_pydm.bool) -> (!iree_pydm.exception_result, !iree_pydm.none) {
  %0 = bool_to_pred %arg0
  %1 = none
  // CHECK: cond_br %arg0
  cond_br %0, ^bb1, ^bb2
^bb1:
  return %1 : !iree_pydm.none
^bb2:
  return %1 : !iree_pydm.none
}

// -----
// CHECK-LABEL: @br
iree_pydm.func @br() -> (!iree_pydm.exception_result, !iree_pydm.none) {
  %0 = none
  // CHECK: br ^bb1({{.*}} : i32)
  br ^bb1(%0 : !iree_pydm.none)
  // CHECK: ^bb1(%0: i32):
^bb1(%1 : !iree_pydm.none):
  return %1 : !iree_pydm.none
}

// -----
// CHECK-LABEL: @box
// NOTE: "78" is the type code for signed i32
iree_pydm.func @box(%arg0 : !iree_pydm.integer<32>) -> (!iree_pydm.exception_result, !iree_pydm.object<!iree_pydm.integer<32>>) {
  // CHECK: %[[LIST:.*]] = iree_input.list.create : !iree_input.list<!iree_input.variant>
  // CHECK: %[[c2:.*]] = arith.constant 2 : index
  // CHECK: iree_input.list.resize %[[LIST]], %c2 : !iree_input.list<!iree_input.variant>
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: %[[c9:.*]] = arith.constant 78 : i32
  // CHECK: iree_input.list.set %[[LIST]][%[[c0]]], %[[c9]] : !iree_input.list<!iree_input.variant>, i32
  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: iree_input.list.set %[[LIST]][%[[c1]]], %arg0 : !iree_input.list<!iree_input.variant>, i32
  // CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
  // return %[[c0_i32]], %[[LIST]] : i32, !iree_input.list<!iree_input.variant>
  %0 = box %arg0 : !iree_pydm.integer<32> -> !iree_pydm.object<!iree_pydm.integer<32>>
  return %0 : !iree_pydm.object<!iree_pydm.integer<32>>
}

// -----
// CHECK-LABEL: @alloc_store_load_var
// NOTE: 256 is the type code for a plain object
iree_pydm.func @alloc_store_load_var(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  // CHECK: %[[A:.*]] = iree_input.list.create : !iree_input.list<!iree_input.variant>
  %a = alloc_free_var "a" -> !iree_pydm.free_var_ref
  // CHECK: %[[c2:.*]] = arith.constant 2 : index
  // CHECK: iree_input.list.resize %[[A]], %[[c2]] : !iree_input.list<!iree_input.variant>
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: %[[object_code:.*]] = arith.constant 256 : i32
  // CHECK: iree_input.list.set %[[A]][%[[c0]]], %[[object_code]]
  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: iree_input.list.set %[[A]][%[[c1]]], %arg0 : !iree_input.list<!iree_input.variant>, !iree_input.list<!iree_input.variant>
  store_var %a = %arg0 : !iree_pydm.free_var_ref, !iree_pydm.object
  // CHECK: %[[c1_0:.*]] = arith.constant 1 : index
  // CHECK: %[[LOADED:.*]] = iree_input.list.get %[[A]][%[[c1_0]]] : !iree_input.list<!iree_input.variant> -> !iree_input.list<!iree_input.variant>
  %0 = load_var %a : !iree_pydm.free_var_ref -> !iree_pydm.object
  // CHECK: return {{.*}}, %[[LOADED]]
  return %0 : !iree_pydm.object
}

// -----
// CHECK-LABEL: @unbox
// NOTE: "78" is the type code for signed i32
iree_pydm.func @unbox(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.integer<32>) {
  // CHECK:  %[[c0:.*]] = arith.constant 0 : index
  // CHECK: %[[NEEDED_TYPE_CODE:.*]] = arith.constant 78 : i32
  // CHECK: %[[TYPE_CODE:.*]] = iree_input.list.get %arg0[%[[c0]]] : !iree_input.list<!iree_input.variant> -> i32
  // CHECK: %[[TYPE_EQ:.*]] = arith.cmpi eq, %[[NEEDED_TYPE_CODE]], %[[TYPE_CODE]] : i32
  // CHECK: cond_br %[[TYPE_EQ]], ^bb1, ^bb4

  // bb1: On equal
  // CHECK: ^bb1:
  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
  // CHECK: %[[CONTENTS:.*]] = iree_input.list.get %arg0[%[[c1]]] : !iree_input.list<!iree_input.variant> -> i32
  // CHECK: br ^bb2(%[[c0_i32]], %[[CONTENTS]] : i32, i32)

  // bb2: Check status code (from raise_on_failure)
  // CHECK: ^bb2(%3: i32, %4: i32):  // 2 preds: ^bb1, ^bb4

  // bb3: Return success
  // CHECK: ^bb3

  // bb4: Signal ValueError (-4 == ValueError)
  // CHECK: ^bb4:
  // CHECK: %[[VALUE_ERROR_CODE:.*]] = arith.constant -4 : i32
  // CHECK: %[[c0_i32_2:.*]] = arith.constant 0 : i32
  // CHECK: br ^bb2(%[[VALUE_ERROR_CODE]], %[[c0_i32_2]] : i32, i32)
  %status, %primitive = unbox %arg0 : !iree_pydm.object -> !iree_pydm.integer<32>
  raise_on_failure %status : !iree_pydm.exception_result
  return %primitive : !iree_pydm.integer<32>
}

// -----
// CHECK-LABEL: @raise_on_failure_object_return
iree_pydm.func @raise_on_failure_object_return(%arg0 : !iree_pydm.exception_result, %arg1: !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  // CHECK: %[[c0_i32:.*]] = arith.constant 0 : i32
  // CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[c0_i32]], %arg0 : i32
  // CHECK: cond_br %[[CMP]], ^bb1, ^bb2
  // bb1: success
  // CHECK: ^bb1:
  // CHECK: %[[c0_i32_0:.*]] = arith.constant 0 : i32
  // CHECK: return %[[c0_i32_0]], %arg1 : i32, !iree_input.list<!iree_input.variant>
  // bb2: failure
  // CHECK: ^bb2:
  // CHECK: %[[NULL:.*]] = iree_input.list.create : !iree_input.list<!iree_input.variant>
  // CHECK: return %arg0, %[[NULL]] : i32, !iree_input.list<!iree_input.variant>
  raise_on_failure %arg0 : !iree_pydm.exception_result
  return %arg1 : !iree_pydm.object
}

// -----
// CHECK-LABEL: @raise_on_failure_builtin
iree_pydm.func @raise_on_failure_builtin(%arg0 : !iree_pydm.exception_result, %arg1: !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  // bb2: failure
  // CHECK: ^bb2:
  // CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
  // CHECK: return %arg0, %[[ZERO]] : i32, i32
  raise_on_failure %arg0 : !iree_pydm.exception_result
  return %arg1 : !iree_pydm.integer
}

// -----
// CHECK-LABEL: @call_and_visibility
iree_pydm.func @call_and_visibility(%arg0 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  // CHECK: %[[R:.*]]:2 = call @callee(%arg0) : (i32) -> (i32, i32)
  %0:2 = call @callee(%arg0) : (!iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.integer)
  return %0#1 : !iree_pydm.integer
}

// CHECK: func private @callee
iree_pydm.func private @callee(%arg0 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  return %arg0 : !iree_pydm.integer
}

// -----
// CHECK-LABEL: @get_type_code
iree_pydm.func @get_type_code(%arg0 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.integer) {
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: %[[R:.*]] = iree_input.list.get %arg0[%[[c0]]] : !iree_input.list<!iree_input.variant> -> i32
  %0 = get_type_code %arg0 : !iree_pydm.object -> !iree_pydm.integer
  return %0 : !iree_pydm.integer
}

// -----
// CHECK-LABEL: @elide_static_info_cast
iree_pydm.func @elide_static_info_cast(%arg0 : !iree_pydm.object<!iree_pydm.integer>) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  // CHECK-NOT: static_info_cast
  %0 = static_info_cast %arg0 : !iree_pydm.object<!iree_pydm.integer> -> !iree_pydm.object
  return %0 : !iree_pydm.object
}

// -----
// CHECK-LABEL:   func @make_tuple(
// CHECK-SAME:                     %[[VAL_0:.*]]: !iree_input.list<!iree_input.variant>,
// CHECK-SAME:                     %[[VAL_1:.*]]: !iree_input.list<!iree_input.variant>) -> (i32, !iree_input.list<!iree_input.variant>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = iree_input.list.create %[[VAL_2]] : !iree_input.list<!iree_input.variant>
// CHECK:           iree_input.list.resize %[[VAL_3]], %[[VAL_2]] : !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           iree_input.list.set %[[VAL_3]]{{\[}}%[[VAL_4]]], %[[VAL_0]] : !iree_input.list<!iree_input.variant>, !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           iree_input.list.set %[[VAL_3]]{{\[}}%[[VAL_5]]], %[[VAL_1]] : !iree_input.list<!iree_input.variant>, !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_6]], %[[VAL_3]] : i32, !iree_input.list<!iree_input.variant>
// CHECK:         }
iree_pydm.func @make_tuple(%arg0 : !iree_pydm.object<!iree_pydm.integer>, %arg1 : !iree_pydm.object<!iree_pydm.integer>) -> (!iree_pydm.exception_result, !iree_pydm.tuple) {
  %0 = make_tuple %arg0, %arg1 : !iree_pydm.object<!iree_pydm.integer>, !iree_pydm.object<!iree_pydm.integer> -> !iree_pydm.tuple
  return %0 : !iree_pydm.tuple
}

// -----
// CHECK-LABEL:   func @make_list(
// CHECK-SAME:                    %[[VAL_0:.*]]: !iree_input.list<!iree_input.variant>,
// CHECK-SAME:                    %[[VAL_1:.*]]: !iree_input.list<!iree_input.variant>) -> (i32, !iree_input.list<!iree_input.variant>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = iree_input.list.create %[[VAL_2]] : !iree_input.list<!iree_input.variant>
// CHECK:           iree_input.list.resize %[[VAL_3]], %[[VAL_2]] : !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           iree_input.list.set %[[VAL_3]]{{\[}}%[[VAL_4]]], %[[VAL_0]] : !iree_input.list<!iree_input.variant>, !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           iree_input.list.set %[[VAL_3]]{{\[}}%[[VAL_5]]], %[[VAL_1]] : !iree_input.list<!iree_input.variant>, !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_6]], %[[VAL_3]] : i32, !iree_input.list<!iree_input.variant>
// CHECK:         }
iree_pydm.func @make_list(%arg0 : !iree_pydm.object<!iree_pydm.integer>, %arg1 : !iree_pydm.object<!iree_pydm.integer>) -> (!iree_pydm.exception_result, !iree_pydm.list) {
  %0 = make_list %arg0, %arg1 : !iree_pydm.object<!iree_pydm.integer>, !iree_pydm.object<!iree_pydm.integer> -> !iree_pydm.list
  return %0 : !iree_pydm.list
}

// -----
// CHECK-LABEL:   func @dynamic_unpack(
// CHECK-SAME:                         %[[VAL_0:.*]]: !iree_input.list<!iree_input.variant>) -> (i32, !iree_input.list<!iree_input.variant>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_2:.*]] = iree_input.list.size %[[VAL_0]] : !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_3:.*]] = arith.cmpi eq, %[[VAL_1]], %[[VAL_2]] : index
// CHECK:           cond_br %[[VAL_3]], ^bb1, ^bb4
// CHECK:         ^bb1:
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = iree_input.list.get %[[VAL_0]]{{\[}}%[[VAL_5]]] : !iree_input.list<!iree_input.variant> -> i32
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = iree_input.list.get %[[VAL_0]]{{\[}}%[[VAL_7]]] : !iree_input.list<!iree_input.variant> -> i1
// CHECK:           br ^bb2(%[[VAL_4]], %[[VAL_6]], %[[VAL_8]] : i32, i32, i1)
// CHECK:         ^bb2(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i1):
// CHECK:           %[[VAL_12:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_13:.*]] = arith.cmpi eq, %[[VAL_12]], %[[VAL_9]] : i32
// CHECK:           cond_br %[[VAL_13]], ^bb3, ^bb5
// CHECK:         ^bb3:
// CHECK:           %[[VAL_14:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_15:.*]] = iree_input.list.create %[[VAL_14]] : !iree_input.list<!iree_input.variant>
// CHECK:           iree_input.list.resize %[[VAL_15]], %[[VAL_14]] : !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:           iree_input.list.set %[[VAL_15]]{{\[}}%[[VAL_16]]], %[[VAL_10]] : !iree_input.list<!iree_input.variant>, i32
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           iree_input.list.set %[[VAL_15]]{{\[}}%[[VAL_17]]], %[[VAL_11]] : !iree_input.list<!iree_input.variant>, i1
// CHECK:           %[[VAL_18:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_18]], %[[VAL_15]] : i32, !iree_input.list<!iree_input.variant>
// CHECK:         ^bb4:
// CHECK:           %[[VAL_19:.*]] = arith.constant -4 : i32
// CHECK:           %[[VAL_20:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_21:.*]] = arith.constant false
// CHECK:           br ^bb2(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]] : i32, i32, i1)
// CHECK:         ^bb5:
// CHECK:           %[[VAL_22:.*]] = iree_input.list.create : !iree_input.list<!iree_input.variant>
// CHECK:           return %[[VAL_9]], %[[VAL_22]] : i32, !iree_input.list<!iree_input.variant>
// CHECK:         }
iree_pydm.func @dynamic_unpack(%arg0 : !iree_pydm.tuple) -> (!iree_pydm.exception_result, !iree_pydm.tuple) {
  %exc_result, %0, %1 = dynamic_unpack %arg0 : !iree_pydm.tuple -> !iree_pydm.exception_result, [!iree_pydm.integer, !iree_pydm.bool]
  raise_on_failure %exc_result : !iree_pydm.exception_result
  %result = make_tuple %0, %1 : !iree_pydm.integer, !iree_pydm.bool -> !iree_pydm.tuple
  return %result : !iree_pydm.tuple
}

// -----
// CHECK-LABEL:   func @list_duplicate(
// CHECK-SAME:                         %[[VAL_0:.*]]: !iree_input.list<!iree_input.variant>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i32) -> (i32, !iree_input.list<!iree_input.variant>) {
// CHECK:           %[[VAL_2:.*]] = iree_input.list.size %[[VAL_0]] : !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_7:.*]] = arith.cmpi sle, %[[VAL_1]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_8:.*]] = select %[[VAL_7]], %[[VAL_4]], %[[VAL_3]] : index
// CHECK:           %[[VAL_9:.*]] = arith.muli %[[VAL_2]], %[[VAL_8]] : index
// CHECK:           %[[VAL_10:.*]] = iree_input.list.create %[[VAL_8]] : !iree_input.list<!iree_input.variant>
// CHECK:           iree_input.list.resize %[[VAL_10]], %[[VAL_9]] : !iree_input.list<!iree_input.variant>
// CHECK:           br ^bb1(%[[VAL_4]] : index)
// CHECK:         ^bb1(%[[VAL_11:.*]]: index):
// CHECK:           %[[VAL_12:.*]] = arith.cmpi ult, %[[VAL_11]], %[[VAL_9]] : index
// CHECK:           cond_br %[[VAL_12]], ^bb2(%[[VAL_11]], %[[VAL_4]] : index, index), ^bb4
// CHECK:         ^bb2(%[[VAL_13:.*]]: index, %[[VAL_14:.*]]: index):
// CHECK:           %[[VAL_15:.*]] = arith.cmpi ult, %[[VAL_14]], %[[VAL_2]] : index
// CHECK:           cond_br %[[VAL_15]], ^bb3(%[[VAL_13]], %[[VAL_14]] : index, index), ^bb1(%[[VAL_13]] : index)
// CHECK:         ^bb3(%[[VAL_16:.*]]: index, %[[VAL_17:.*]]: index):
// CHECK:           %[[VAL_18:.*]] = iree_input.list.get %[[VAL_0]]{{\[}}%[[VAL_17]]] : !iree_input.list<!iree_input.variant> -> !iree_input.list<!iree_input.variant>
// CHECK:           iree_input.list.set %[[VAL_10]]{{\[}}%[[VAL_16]]], %[[VAL_18]] : !iree_input.list<!iree_input.variant>, !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_16]], %[[VAL_5]] : index
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_17]], %[[VAL_5]] : index
// CHECK:           br ^bb2(%[[VAL_19]], %[[VAL_20]] : index, index)
// CHECK:         ^bb4:
// CHECK:           %[[VAL_21:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_21]], %[[VAL_10]] : i32, !iree_input.list<!iree_input.variant>
// CHECK:         }
iree_pydm.func @list_duplicate(%arg0 : !iree_pydm.list, %arg1 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.list) {
  %result = sequence_clone %arg0 * %arg1 : !iree_pydm.list, !iree_pydm.integer -> !iree_pydm.list
  return %result : !iree_pydm.list
}

// -----
// CHECK-LABEL:   func @subscript_list(
// CHECK-SAME:                         %[[VAL_0:.*]]: !iree_input.list<!iree_input.variant>,
// CHECK-SAME:                         %[[VAL_1:.*]]: i32) -> (i32, !iree_input.list<!iree_input.variant>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_3:.*]] = iree_input.list.size %[[VAL_0]] : !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : index to i32
// CHECK:           %[[VAL_5:.*]] = arith.cmpi slt, %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           cond_br %[[VAL_5]], ^bb1, ^bb2(%[[VAL_6]] : index)
// CHECK:         ^bb1:
// CHECK:           %[[VAL_7:.*]] = arith.addi %[[VAL_1]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]] : i32 to index
// CHECK:           br ^bb2(%[[VAL_8]] : index)
// CHECK:         ^bb2(%[[VAL_9:.*]]: index):
// CHECK:           %[[VAL_10:.*]] = arith.cmpi ult, %[[VAL_9]], %[[VAL_3]] : index
// CHECK:           cond_br %[[VAL_10]], ^bb3(%[[VAL_9]] : index), ^bb6
// CHECK:         ^bb3(%[[VAL_11:.*]]: index):
// CHECK:           %[[VAL_12:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_13:.*]] = iree_input.list.get %[[VAL_0]]{{\[}}%[[VAL_11]]] : !iree_input.list<!iree_input.variant> -> !iree_input.list<!iree_input.variant>
// CHECK:           br ^bb4(%[[VAL_12]], %[[VAL_13]] : i32, !iree_input.list<!iree_input.variant>)
// CHECK:         ^bb4(%[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: !iree_input.list<!iree_input.variant>):
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_17:.*]] = arith.cmpi eq, %[[VAL_16]], %[[VAL_14]] : i32
// CHECK:           cond_br %[[VAL_17]], ^bb5, ^bb7
// CHECK:         ^bb5:
// CHECK:           %[[VAL_18:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_18]], %[[VAL_15]] : i32, !iree_input.list<!iree_input.variant>
// CHECK:         ^bb6:
// CHECK:           %[[VAL_19:.*]] = arith.constant -7 : i32
// CHECK:           %[[VAL_20:.*]] = iree_input.list.create : !iree_input.list<!iree_input.variant>
// CHECK:           br ^bb4(%[[VAL_19]], %[[VAL_20]] : i32, !iree_input.list<!iree_input.variant>)
// CHECK:         ^bb7:
// CHECK:           %[[VAL_21:.*]] = iree_input.list.create : !iree_input.list<!iree_input.variant>
// CHECK:           return %[[VAL_14]], %[[VAL_21]] : i32, !iree_input.list<!iree_input.variant>
// CHECK:         }
iree_pydm.func @subscript_list(%arg0 : !iree_pydm.list, %arg1 : !iree_pydm.integer) -> (!iree_pydm.exception_result, !iree_pydm.object) {
  %exc_result, %result = subscript %arg0[%arg1] : !iree_pydm.list, !iree_pydm.integer -> !iree_pydm.object
  raise_on_failure %exc_result : !iree_pydm.exception_result
  return %result : !iree_pydm.object
}

// -----
// CHECK-LABEL:   func @assign_subscript_list(
// CHECK-SAME:                                %[[VAL_0:.*]]: !iree_input.list<!iree_input.variant>,
// CHECK-SAME:                                %[[VAL_1:.*]]: i32,
// CHECK-SAME:                                %[[VAL_2:.*]]: !iree_input.list<!iree_input.variant>) -> (i32, !iree_input.list<!iree_input.variant>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = iree_input.list.size %[[VAL_0]] : !iree_input.list<!iree_input.variant>
// CHECK:           %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : index to i32
// CHECK:           %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_1]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_7:.*]] = arith.index_cast %[[VAL_1]] : i32 to index
// CHECK:           cond_br %[[VAL_6]], ^bb1, ^bb2(%[[VAL_7]] : index)
// CHECK:         ^bb1:
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_1]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[VAL_8]] : i32 to index
// CHECK:           br ^bb2(%[[VAL_9]] : index)
// CHECK:         ^bb2(%[[VAL_10:.*]]: index):
// CHECK:           %[[VAL_11:.*]] = arith.cmpi ult, %[[VAL_10]], %[[VAL_4]] : index
// CHECK:           cond_br %[[VAL_11]], ^bb3(%[[VAL_10]] : index), ^bb5
// CHECK:         ^bb3(%[[VAL_12:.*]]: index):
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i32
// CHECK:           iree_input.list.set %[[VAL_0]]{{\[}}%[[VAL_12]]], %[[VAL_2]] : !iree_input.list<!iree_input.variant>, !iree_input.list<!iree_input.variant>
// CHECK:           br ^bb4(%[[VAL_13]] : i32)
// CHECK:         ^bb4(%[[VAL_14:.*]]: i32):
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_15]], %[[VAL_0]] : i32, !iree_input.list<!iree_input.variant>
// CHECK:         ^bb5:
// CHECK:           %[[VAL_16:.*]] = arith.constant -7 : i32
// CHECK:           br ^bb4(%[[VAL_16]] : i32)
// CHECK:         }
iree_pydm.func @assign_subscript_list(%arg0 : !iree_pydm.list, %arg1 : !iree_pydm.integer, %arg2 : !iree_pydm.object) -> (!iree_pydm.exception_result, !iree_pydm.list) {
  assign_subscript %arg0[%arg1] = %arg2 : !iree_pydm.list, !iree_pydm.integer, !iree_pydm.object
  return %arg0 : !iree_pydm.list
}
