// RUN: iree-opt --iree-util-lift-cfg-to-scf --split-input-file %s | FileCheck %s

// Tests simple if conversion.

util.func public @simple_if(%cond: i1) {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  util.buffer.constant : !util.buffer = "bb1"
  cf.br ^bb3
^bb2:
  util.buffer.constant : !util.buffer = "bb2"
  cf.br ^bb3
^bb3:
  util.return
}
// CHECK-LABEL: util.func public @simple_if
// CHECK: scf.if %{{.*}} {
// CHECK:   !util.buffer = "bb1"
// CHECK: } else {
// CHECK:   !util.buffer = "bb2"
// CHECK: }
// CHECK: util.return

// -----

// Tests loop conversion.

util.func public @simple_loop(%count: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  cf.br ^bb1(%c0: index)
^bb1(%i: index):
  %cond = arith.cmpi slt, %i, %count : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  util.null : !util.buffer
  %next = arith.addi %i, %c1 : index
  cf.br ^bb1(%next: index)
^bb3:
  util.return
}
// CHECK-LABEL: util.func public @simple_loop
// CHECK: scf.while
// CHECK:   util.null : !util.buffer
// CHECK:   scf.condition
// CHECK: } do {
// CHECK:   scf.yield
// CHECK: }
// CHECK: util.return

// -----

// Tests unreachable handling for infinite loops.

util.func public @infinite_loop() {
  cf.br ^bb1
^bb1:
  util.null : !util.buffer
  cf.br ^bb1
}
// CHECK-LABEL: util.func public @infinite_loop
// CHECK: scf.while
// CHECK: util.scf.unreachable "infinite loop"

// -----

// Tests multiple returns.

util.func public @multiple_returns(%cond: i1) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  util.return %c1 : i32
^bb2:
  %c2 = arith.constant 2 : i32
  util.return %c2 : i32
}
// CHECK-LABEL: util.func public @multiple_returns
// CHECK: %[[RESULT:.+]] = scf.if %{{.*}} -> (i32) {
// CHECK:   %[[C1:.+]] = arith.constant 1
// CHECK:   scf.yield %[[C1]]
// CHECK: } else {
// CHECK:   %[[C2:.+]] = arith.constant 2
// CHECK:   scf.yield %[[C2]]
// CHECK: }
// CHECK: util.return %[[RESULT]]

// -----

// Tests a util.initializer with control flow.

util.global private mutable @global1 : i32
util.global private mutable @global2 : i32

util.initializer {
  // %cond_value = arith.constant true
  %cond = arith.constant true
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %c1 = arith.constant 1 : i32
  util.global.store %c1, @global1 : i32
  cf.br ^bb3
^bb2:
  %c2 = arith.constant 2 : i32
  util.global.store %c2, @global2 : i32
  cf.br ^bb3
^bb3:
  util.return
}
// CHECK-LABEL: util.initializer
// CHECK: %[[COND:.+]] = arith.constant true
// CHECK: scf.if %[[COND]] {
// CHECK:   %[[C1:.+]] = arith.constant 1
// CHECK:   util.global.store %[[C1]]
// CHECK: } else {
// CHECK:   %[[C2:.+]] = arith.constant 2
// CHECK:   util.global.store %[[C2]]
// CHECK: }
// CHECK: util.return

// -----

// Tests nested control flow.

util.func public @nested_if(%cond1: i1, %cond2: i1) -> i32 {
  cf.cond_br %cond1, ^bb1, ^bb4
^bb1:
  cf.cond_br %cond2, ^bb2, ^bb3
^bb2:
  %c1 = arith.constant 1 : i32
  cf.br ^bb5(%c1 : i32)
^bb3:
  %c2 = arith.constant 2 : i32
  cf.br ^bb5(%c2 : i32)
^bb4:
  %c3 = arith.constant 3 : i32
  cf.br ^bb5(%c3 : i32)
^bb5(%result: i32):
  util.return %result : i32
}
// CHECK-LABEL: util.func public @nested_if
// CHECK-SAME: (%[[COND1:.+]]: i1, %[[COND2:.+]]: i1)
// CHECK: %[[RESULT:.+]] = scf.if %[[COND1]] -> (i32) {
// CHECK:   %[[INNER:.+]] = scf.if %[[COND2]] -> (i32) {
// CHECK:     %[[C1:.+]] = arith.constant 1
// CHECK:     scf.yield %[[C1]]
// CHECK:   } else {
// CHECK:     %[[C2:.+]] = arith.constant 2
// CHECK:     scf.yield %[[C2]]
// CHECK:   }
// CHECK:   scf.yield %[[INNER]]
// CHECK: } else {
// CHECK:   %[[C3:.+]] = arith.constant 3
// CHECK:   scf.yield %[[C3]]
// CHECK: }
// CHECK: util.return %[[RESULT]]

// -----

// Tests a loop with values passed through iterations.

util.func public @loop_with_values(%n: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  cf.br ^bb1(%c0 : index)
^bb1(%sum: index):
  %cond = arith.cmpi slt, %sum, %n : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %next = arith.addi %sum, %c1 : index
  cf.br ^bb1(%next : index)
^bb3:
  util.return %sum : index
}
// CHECK-LABEL: util.func public @loop_with_values
// CHECK-SAME: (%[[ARG0:.+]]: index)
// CHECK: %[[POISON:.+]] = ub.poison
// CHECK: %[[RESULT:.+]]:2 = scf.while (%[[ARG1:.+]] = %c0{{.*}}, %[[ARG2:.+]] = %{{.+}}) : (index, index) -> (index, index) {
// CHECK:   %[[COND:.+]] = arith.cmpi slt, %[[ARG1]], %[[ARG0]]
// CHECK:   %[[PATH:.+]]:3 = scf.if %[[COND]] -> (index, index, index) {
// CHECK:     %[[NEXT:.+]] = arith.addi %[[ARG1]], %c1
// CHECK:     scf.yield %[[NEXT]], %c0{{.*}}, %c1{{.*}}
// CHECK:   } else {
// CHECK:     scf.yield %[[POISON]], %c1{{.*}}, %c0{{.*}}
// CHECK:   }
// CHECK:   arith.index_castui %[[PATH]]#2 : index to i64
// CHECK:   %[[PATH_I1:.+]] = arith.trunci %{{.+}} : i64 to i1
// CHECK:   scf.condition(%[[PATH_I1]]) %[[PATH]]#0, %[[ARG1]]
// CHECK: } do {
// CHECK: ^bb0(%[[BB_ARG1:.+]]: index, %[[BB_ARG2:.+]]: index):
// CHECK:   scf.yield %[[BB_ARG1]], %[[BB_ARG2]]
// CHECK: }
// CHECK: util.return %[[RESULT]]#1

// -----

// Tests nested if inside while loop.
//
// Pseudo-code:
//   index i = 0;
//   while (i < n) {
//     if (i % 2 == 0) {
//       i += 1;
//     } else {
//       i += 2;
//     }
//   }
//   return i;

util.func public @if_in_while(%n: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  cf.br ^loop(%c0 : index)
^loop(%i: index):
  %cond = arith.cmpi slt, %i, %n : index
  cf.cond_br %cond, ^body, ^exit
^body:
  // Nested if inside loop body.
  %is_even = arith.remui %i, %c2 : index
  %is_zero = arith.cmpi eq, %is_even, %c0 : index
  cf.cond_br %is_zero, ^even, ^odd
^even:
  %inc1 = arith.addi %i, %c1 : index
  cf.br ^continue(%inc1 : index)
^odd:
  %inc2 = arith.addi %i, %c2 : index
  cf.br ^continue(%inc2 : index)
^continue(%next: index):
  cf.br ^loop(%next : index)
^exit:
  util.return %i : index
}
// CHECK-LABEL: util.func public @if_in_while
// CHECK: scf.while
// CHECK:   scf.if
// CHECK:     scf.yield
// CHECK:   } else {
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.condition
// CHECK: } do {
// CHECK:   scf.yield
// CHECK: }

// -----

// Tests while loops inside if branches.
//
// Pseudo-code:
//   if (cond) {
//     index i = 0;
//     while (i < n) {
//       i += 1;
//     }
//     return i;
//   } else {
//     index j = 10;
//     while (j > n) {
//       j -= 2;
//     }
//     return j;
//   }

util.func public @while_in_if(%cond: i1, %n: index) -> index {
  cf.cond_br %cond, ^branch1, ^branch2
^branch1:
  // While loop in first branch.
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  cf.br ^loop1(%c0 : index)
^loop1(%i: index):
  %done1 = arith.cmpi slt, %i, %n : index
  cf.cond_br %done1, ^body1, ^end1
^body1:
  %next1 = arith.addi %i, %c1 : index
  cf.br ^loop1(%next1 : index)
^end1:
  cf.br ^exit(%i : index)
^branch2:
  // Different while loop in second branch.
  %c10 = arith.constant 10 : index
  %c2 = arith.constant 2 : index
  cf.br ^loop2(%c10 : index)
^loop2(%j: index):
  %done2 = arith.cmpi sgt, %j, %n : index
  cf.cond_br %done2, ^body2, ^end2
^body2:
  %next2 = arith.subi %j, %c2 : index
  cf.br ^loop2(%next2 : index)
^end2:
  cf.br ^exit(%j : index)
^exit(%result: index):
  util.return %result : index
}
// CHECK-LABEL: util.func public @while_in_if
// CHECK: scf.if %{{.*}} -> (index) {
// CHECK:   scf.while
// CHECK:     scf.condition
// CHECK:   } do {
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield
// CHECK: } else {
// CHECK:   scf.while
// CHECK:     scf.condition
// CHECK:   } do {
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield
// CHECK: }

// -----

// Tests that existing SCF ops are preserved and not modified.
//
// Pseudo-code:
//   index existing_if = (cond ? 5 : 10);
//   index existing_while = existing_if;
//   while (existing_while < n) {
//     existing_while += 1;
//   }
//   if (cond) {
//     return existing_while + existing_if;
//   } else {
//     return existing_while - existing_if;
//   }

util.func public @preserve_existing_scf(%cond: i1, %n: index) -> index {
  // This scf.if should be preserved as-is.
  %existing_if = scf.if %cond -> index {
    %c5 = arith.constant 5 : index
    scf.yield %c5 : index
  } else {
    %c10 = arith.constant 10 : index
    scf.yield %c10 : index
  }

  // This scf.while should be preserved as-is.
  %existing_while = scf.while (%iter = %existing_if) : (index) -> index {
    %continue = arith.cmpi slt, %iter, %n : index
    scf.condition(%continue) %iter : index
  } do {
  ^bb0(%arg: index):
    %c1 = arith.constant 1 : index
    %next = arith.addi %arg, %c1 : index
    scf.yield %next : index
  }

  // Now some CFG that needs conversion.
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %add = arith.addi %existing_while, %existing_if : index
  cf.br ^exit(%add : index)
^bb2:
  %sub = arith.subi %existing_while, %existing_if : index
  cf.br ^exit(%sub : index)
^exit(%result: index):
  util.return %result : index
}
// CHECK-LABEL: util.func public @preserve_existing_scf
// CHECK: %[[EXISTING_IF:.+]] = scf.if %{{.*}} -> (index) {
// CHECK:   %[[C5:.+]] = arith.constant 5
// CHECK:   scf.yield %[[C5]]
// CHECK: } else {
// CHECK:   %[[C10:.+]] = arith.constant 10
// CHECK:   scf.yield %[[C10]]
// CHECK: }
// CHECK: %[[EXISTING_WHILE:.+]] = scf.while (%{{.*}} = %[[EXISTING_IF]]) : (index) -> index {
// CHECK:   arith.cmpi slt
// CHECK:   scf.condition
// CHECK: } do {
// CHECK: ^bb{{.*}}(%{{.*}}: index):
// CHECK:   arith.constant 1
// CHECK:   arith.addi
// CHECK:   scf.yield
// CHECK: }
// CHECK: scf.if %{{.*}} -> (index) {
// CHECK:   arith.addi %[[EXISTING_WHILE]], %[[EXISTING_IF]]
// CHECK:   scf.yield
// CHECK: } else {
// CHECK:   arith.subi %[[EXISTING_WHILE]], %[[EXISTING_IF]]
// CHECK:   scf.yield
// CHECK: }

// -----

// Tests mixed CFG and SCF - SCF ops inside CFG blocks.
//
// Pseudo-code:
//   if (cond) {
//     index inner_result = (cond ? 1 : 2);
//     return inner_result;
//   } else {
//     index i = 0;
//     while (i < n) {
//       i += 1;
//     }
//     return i;
//   }

util.func public @mixed_scf_in_cfg(%cond: i1, %n: index) -> index {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  // SCF.if inside a CFG block.
  %inner_result = scf.if %cond -> index {
    %c1 = arith.constant 1 : index
    scf.yield %c1 : index
  } else {
    %c2 = arith.constant 2 : index
    scf.yield %c2 : index
  }
  cf.br ^bb3(%inner_result : index)
^bb2:
  // SCF.while inside another CFG block.
  %c0 = arith.constant 0 : index
  %loop_result = scf.while (%i = %c0) : (index) -> index {
    %continue = arith.cmpi slt, %i, %n : index
    scf.condition(%continue) %i : index
  } do {
  ^bb0(%iter: index):
    %c1 = arith.constant 1 : index
    %next = arith.addi %iter, %c1 : index
    scf.yield %next : index
  }
  cf.br ^bb3(%loop_result : index)
^bb3(%final: index):
  util.return %final : index
}
// CHECK-LABEL: util.func public @mixed_scf_in_cfg
// CHECK: scf.if %{{.*}} -> (index) {
// CHECK:   scf.if %{{.*}} -> (index) {
// CHECK:     arith.constant 1
// CHECK:     scf.yield
// CHECK:   } else {
// CHECK:     arith.constant 2
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield
// CHECK: } else {
// CHECK:   arith.constant 0
// CHECK:   scf.while
// CHECK:     scf.condition
// CHECK:   } do {
// CHECK:     arith.constant 1
// CHECK:     arith.addi
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield
// CHECK: }

// -----

// Tests deeply nested control flow (if in while in if).
//
// Pseudo-code:
//   if (cond1) {
//     index i = 0;
//     while (i < n) {
//       if (cond2) {
//         i += 1;
//       } else {
//         i += 2;
//       }
//     }
//     return i;
//   } else {
//     return 100;
//   }

util.func public @deeply_nested(%cond1: i1, %cond2: i1, %n: index) -> index {
  cf.cond_br %cond1, ^outer_then, ^outer_else
^outer_then:
  %c0 = arith.constant 0 : index
  cf.br ^loop_start(%c0 : index)
^loop_start(%i: index):
  %loop_cond = arith.cmpi slt, %i, %n : index
  cf.cond_br %loop_cond, ^loop_body, ^loop_end
^loop_body:
  cf.cond_br %cond2, ^inner_then, ^inner_else
^inner_then:
  %c1 = arith.constant 1 : index
  %inc1 = arith.addi %i, %c1 : index
  cf.br ^loop_continue(%inc1 : index)
^inner_else:
  %c2 = arith.constant 2 : index
  %inc2 = arith.addi %i, %c2 : index
  cf.br ^loop_continue(%inc2 : index)
^loop_continue(%next: index):
  cf.br ^loop_start(%next : index)
^loop_end:
  cf.br ^final(%i : index)
^outer_else:
  %c100 = arith.constant 100 : index
  cf.br ^final(%c100 : index)
^final(%result: index):
  util.return %result : index
}
// CHECK-LABEL: util.func public @deeply_nested
// CHECK: scf.if %{{.*}} -> (index) {
// CHECK:   scf.while
// CHECK:     scf.if %{{.*}} -> (index) {
// CHECK:       arith.constant 1
// CHECK:       arith.addi
// CHECK:       scf.yield
// CHECK:     } else {
// CHECK:       arith.constant 2
// CHECK:       arith.addi
// CHECK:       scf.yield
// CHECK:     }
// CHECK:     scf.condition
// CHECK:   } do {
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield
// CHECK: } else {
// CHECK:   arith.constant 100
// CHECK:   scf.yield
// CHECK: }

// -----

// Tests external functions are skipped.

util.func private @external_func(%arg: i32) -> i32

// -----

// Tests cf.switch with fallthrough patterns.
// Adapted from upstream test: @switch_with_fallthrough
//
// Pseudo-code:
//   switch (flag) {
//     case 0:
//       goto case_shared(arg2);
//     case 1:
//       goto exit;
//     default:
//       float res = callee_a(arg1);
//       goto case_shared(res);
//   }
// case_shared(value):
//   callee_b(value);
//   goto exit;
// exit:
//   return;

util.func private @callee_a(%arg: f32) -> f32
util.func private @callee_b(%arg: f32)

util.func public @switch_with_fallthrough(%flag: i32, %arg1 : f32, %arg2 : f32) {
  cf.switch %flag : i32, [
    default: ^bb1(%arg1 : f32),
    0: ^bb2(%arg2 : f32),
    1: ^bb3
  ]

^bb1(%arg3 : f32):
  %0 = util.call @callee_a(%arg3) : (f32) -> f32
  cf.br ^bb2(%0 : f32)

^bb2(%arg4 : f32):
  util.call @callee_b(%arg4) : (f32) -> ()
  cf.br ^bb3

^bb3:
  util.return
}
// CHECK-LABEL: util.func public @switch_with_fallthrough
// CHECK-SAME: (%[[FLAG:.+]]: i32, %[[ARG1:.+]]: f32, %[[ARG2:.+]]: f32)
// CHECK: %[[INDEX_FLAG:.+]] = arith.index_cast %[[FLAG]] : i32 to index
// CHECK: %[[SWITCH_RESULT:.+]]:2 = scf.index_switch %[[INDEX_FLAG]] -> f32, index
// CHECK: case 0 {
// CHECK:   scf.yield %[[ARG2]], %c0{{.*}} : f32, index
// CHECK: }
// CHECK: case 1 {
// CHECK:   scf.yield %{{.*}}, %c1{{.*}} : f32, index
// CHECK: }
// CHECK: default {
// CHECK:   %[[RES:.+]] = util.call @callee_a(%[[ARG1]])
// CHECK:   scf.yield %[[RES]], %c0{{.*}} : f32, index
// CHECK: }
// Second switch to handle the fallthrough logic:
// CHECK: scf.index_switch
// CHECK: case 0 {
// CHECK:   util.call @callee_b(%[[SWITCH_RESULT]]#0)
// CHECK:   scf.yield
// CHECK: }
// CHECK: default {
// CHECK: }
// CHECK: util.return

// -----

// Tests CFG transformation inside nested regions.
// Adapted from upstream test: @nested_region
//
// Pseudo-code:
//   execute_region {
//     bool cond = test1();
//     if (cond) {
//       test2();
//     } else {
//       test3();
//     }
//     test4();
//   }

util.func public @nested_region() {
  scf.execute_region {
    %c_true = arith.constant true
    %cond = util.optimization_barrier %c_true : i1
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %c1 = arith.constant 1 : i32
    util.optimization_barrier %c1 : i32
    cf.br ^bb3
  ^bb2:
    %c2 = arith.constant 2 : i32
    util.optimization_barrier %c2 : i32
    cf.br ^bb3
  ^bb3:
    %c3 = arith.constant 3 : i32
    util.optimization_barrier %c3 : i32
    scf.yield
  }
  util.return
}
// CHECK-LABEL: util.func public @nested_region
// CHECK: scf.execute_region {
// CHECK:   %[[C_TRUE:.+]] = arith.constant true
// CHECK:   %[[COND:.+]] = util.optimization_barrier %[[C_TRUE]] : i1
// CHECK:   scf.if %[[COND]] {
// CHECK:     %[[C1:.+]] = arith.constant 1
// CHECK:     util.optimization_barrier %[[C1]]
// CHECK:   } else {
// CHECK:     %[[C2:.+]] = arith.constant 2
// CHECK:     util.optimization_barrier %[[C2]]
// CHECK:   }
// CHECK:   %[[C3:.+]] = arith.constant 3
// CHECK:   util.optimization_barrier %[[C3]]
// CHECK:   scf.yield
// CHECK: }
// CHECK: util.return

// -----

// Tests deeply nested regions with multiple levels of nesting.
// This ensures CFG lifting works recursively through multiple region levels.
//
// Pseudo-code:
//   if (outer_cond) {
//     execute_region {
//       execute_region {
//         if (inner_cond) {
//           test1();
//         } else {
//           test2();
//         }
//       }
//     }
//   }

util.func public @deeply_nested_regions(%outer_cond: i1, %inner_cond: i1) {
  cf.cond_br %outer_cond, ^outer_then, ^outer_else
^outer_then:
  scf.execute_region {
    scf.execute_region {
      cf.cond_br %inner_cond, ^inner_then, ^inner_else
    ^inner_then:
      %c42 = arith.constant 42 : i32
      util.optimization_barrier %c42 : i32
      cf.br ^inner_end
    ^inner_else:
      %c24 = arith.constant 24 : i32
      util.optimization_barrier %c24 : i32
      cf.br ^inner_end
    ^inner_end:
      scf.yield
    }
    scf.yield
  }
  cf.br ^outer_end
^outer_else:
  %c100 = arith.constant 100 : i32
  util.optimization_barrier %c100 : i32
  cf.br ^outer_end
^outer_end:
  util.return
}
// CHECK-LABEL: util.func public @deeply_nested_regions
// CHECK: scf.if %{{.*}} {
// CHECK:   scf.execute_region {
// CHECK:     scf.execute_region {
// CHECK:       scf.if %{{.*}} {
// CHECK:         %[[C42:.+]] = arith.constant 42
// CHECK:         util.optimization_barrier %[[C42]]
// CHECK:       } else {
// CHECK:         %[[C24:.+]] = arith.constant 24
// CHECK:         util.optimization_barrier %[[C24]]
// CHECK:       }
// CHECK:       scf.yield
// CHECK:     }
// CHECK:     scf.yield
// CHECK:   }
// CHECK: } else {
// CHECK:   %[[C100:.+]] = arith.constant 100
// CHECK:   util.optimization_barrier %[[C100]]
// CHECK: }
// CHECK: util.return

// -----

// Tests already structured loop interaction.
// Adapted from upstream test: @already_structured_loop
//
// Pseudo-code:
//   do {
//     bool exit = bar(arg);
//     if (exit) break;
//   } while (true);
//   return;

util.func private @check_condition(%arg: f32) -> i1

util.func public @already_structured_loop(%arg: f32) {
  cf.br ^bb0
^bb0:
  %exit = util.call @check_condition(%arg) : (f32) -> i1
  cf.cond_br %exit, ^bb1, ^bb0
^bb1:
  util.return
}
// CHECK-LABEL: util.func public @already_structured_loop
// CHECK-DAG: %[[C1_INDEX:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0_INDEX:.+]] = arith.constant 0 : index
// CHECK: scf.while : () -> () {
// CHECK:   %[[EXIT:.+]] = util.call @check_condition
// CHECK:   %[[COND_PAIR:.+]]:2 = scf.if %[[EXIT]] -> (index, index) {
// CHECK:     scf.yield %[[C1_INDEX]], %[[C0_INDEX]]
// CHECK:   } else {
// CHECK:     scf.yield %[[C0_INDEX]], %[[C1_INDEX]]
// CHECK:   }
// CHECK:   arith.index_castui %[[COND_PAIR]]#1 : index to i64
// CHECK:   %[[CONTINUE:.+]] = arith.trunci %{{.*}} : i64 to i1
// CHECK:   scf.condition(%[[CONTINUE]])
// CHECK: } do {
// CHECK:   scf.yield
// CHECK: }
// CHECK: util.return

// -----

// Tests conditional infinite loop with unreachable insertion.
// Adapted from upstream test: @conditional_infinite_loop
//
// Pseudo-code:
//   if (cond) {
//     while (true) {
//       arg = bar(arg);
//     }
//     // unreachable
//   } else {
//     arg = bar(arg);
//   }
//   return arg;

util.func private @callee(%arg: f32) -> f32

util.func public @conditional_infinite_loop(%arg: f32, %cond: i1) -> f32 {
  cf.cond_br %cond, ^bb1(%arg : f32), ^bb3(%arg : f32)
^bb1(%arg1 : f32):
  %0 = util.call @callee(%arg1) : (f32) -> f32
  cf.br ^bb1(%0 : f32)
^bb3(%arg2 : f32):
  %1 = util.call @callee(%arg2) : (f32) -> f32
  cf.br ^bb4
^bb4:
  util.return %1 : f32
}
// CHECK-LABEL: util.func public @conditional_infinite_loop
// CHECK-SAME: %[[ARG:.+]]: f32, %[[COND:.+]]: i1
// CHECK: %[[RES:.+]] = scf.if %[[COND]] -> (f32) {
// CHECK:   scf.while (%[[ITER:.+]] = %[[ARG]]) : (f32) -> f32 {
// CHECK:     %[[CALL:.+]] = util.call @callee(%[[ITER]])
// CHECK:     arith.index_castui %c1{{.*}} : index to i64
// CHECK:     %{{.*}} = arith.trunci %{{.*}} : i64 to i1
// CHECK:     scf.condition(%{{.*}}) %[[CALL]]
// CHECK:   } do {
// CHECK:   ^bb{{.*}}(%[[VAL:.+]]: f32):
// CHECK:     scf.yield %[[VAL]]
// CHECK:   }
// CHECK:   util.scf.unreachable "infinite loop"
// CHECK:   %[[POISON:.+]] = ub.poison : f32
// CHECK:   scf.yield %[[POISON]]
// CHECK: } else {
// CHECK:   %[[CALL2:.+]] = util.call @callee(%[[ARG]])
// CHECK:   scf.yield %[[CALL2]]
// CHECK: }
// CHECK: util.return %[[RES]]

// -----

// Tests irreducible control flow (multi-entry loop).
// Adapted from upstream test: @multi_entry_loop
//
// NOTE: The upstream algorithm can handle this "multi-entry" pattern
// using multiplexers to transform it into structured control flow, so we expect
// this to pass.
//
// Pseudo-code (cannot be expressed in structured form):
//   if (cond) goto loop_entry1;
//   else goto loop_entry2;
// loop_entry1:
//   exit = comp1(6);
//   if (exit) goto end(6);
//   else goto loop_entry2;
// loop_entry2:
//   exit2 = comp2(5);
//   if (exit2) goto end(5);
//   else goto loop_entry1;  // Creates cycle with two entries
// end(value):
//   process(value);
//   return;

util.func private @comp1(%arg: i32) -> i1
util.func private @comp2(%arg: i32) -> i1
util.func private @process(%arg: i32)

util.func public @multi_entry_loop(%cond: i1) {
  %c6 = arith.constant 6 : i32
  %c5 = arith.constant 5 : i32
  cf.cond_br %cond, ^bb0, ^bb1
^bb0:
  %exit = util.call @comp1(%c6) : (i32) -> i1
  cf.cond_br %exit, ^bb2(%c6 : i32), ^bb1
^bb1:
  %exit2 = util.call @comp2(%c5) : (i32) -> i1
  cf.cond_br %exit2, ^bb2(%c5 : i32), ^bb0
^bb2(%arg3 : i32):
  util.call @process(%arg3) : (i32) -> ()
  util.return
}
// CHECK-LABEL: util.func public @multi_entry_loop
// CHECK: %[[C6:.+]] = arith.constant 6 : i32
// CHECK: %[[C5:.+]] = arith.constant 5 : i32
// The initial multiplexer selection:
// CHECK: %{{.*}}:2 = scf.if %{{.*}} -> (index, i32) {
// CHECK:   scf.yield %c1{{.*}}, %{{.*}} : index, i32
// CHECK: } else {
// CHECK:   scf.yield %c0{{.*}}, %{{.*}} : index, i32
// CHECK: }
// The main loop with multiplexer switch:
// CHECK: %{{.*}}:2 = scf.while (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) : (index, i32) -> (index, i32) {
// CHECK:   scf.index_switch %{{.*}} -> index, i32, index, index
// CHECK:   case 0 {
// CHECK:     %[[EXIT2:.+]] = util.call @comp2(%[[C5]])
// CHECK:     scf.if %[[EXIT2]] -> (index, i32, index, index) {
// CHECK:       scf.yield %{{.*}}, %[[C5]], %c1{{.*}}, %c0{{.*}}
// CHECK:     } else {
// CHECK:       scf.yield %c1{{.*}}, %{{.*}}, %c0{{.*}}, %c1{{.*}}
// CHECK:     }
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   default {
// CHECK:     %[[EXIT1:.+]] = util.call @comp1(%[[C6]])
// CHECK:     scf.if %[[EXIT1]] -> (index, i32, index, index) {
// CHECK:       scf.yield %{{.*}}, %[[C6]], %c1{{.*}}, %c0{{.*}}
// CHECK:     } else {
// CHECK:       scf.yield %c0{{.*}}, %{{.*}}, %c0{{.*}}, %c1{{.*}}
// CHECK:     }
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   arith.index_castui %{{.*}}#3 : index to i64
// CHECK:   %{{.*}} = arith.trunci %{{.*}} : i64 to i1
// CHECK:   scf.condition(%{{.*}}) %{{.*}}#0, %{{.*}}#1
// CHECK: } do {
// CHECK: ^bb{{.*}}(%{{.*}}: index, %{{.*}}: i32):
// CHECK:   scf.yield %{{.*}}, %{{.*}}
// CHECK: }
// CHECK: util.call @process(%{{.*}}#1)
// CHECK: util.return

// -----

// Tests flow.func with empty region (getCallableRegion() returns nullptr).
// This is the specific case that caused the crash in https://github.com/iree-org/iree/issues/22971.
// flow.func's getCallableRegion() returns nullptr, which should be handled
// gracefully by the pass.

// CHECK-LABEL: flow.func private @flow_func_empty_region
module {
  flow.func private @flow_func_empty_region() {
  }
}