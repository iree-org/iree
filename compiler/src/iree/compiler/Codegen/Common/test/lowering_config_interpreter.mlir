// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-codegen-materialize-tuning-specs, iree-codegen-materialize-user-configs, func.func(iree-codegen-lowering-config-interpreter))" \
// RUN:   --iree-codegen-tuning-spec-path=%p/external_strategy_spec.mlir --split-input-file | FileCheck %s

#config = #iree_gpu.lowering_config<{lowering_strategy = "print_me"}>
module attributes { transform.with_named_sequence } {
  func.func @single_config(%arg0: i32) -> i32 {
    %add = arith.addi %arg0, %arg0 {lowering_config = #config} : i32
    return %add : i32
  }

  transform.named_sequence @print_me(%op: !transform.any_op {transform.readonly}) {
    transform.print %op : !transform.any_op
    transform.yield
  }
}

//      CHECK: IR printer:
// CHECK-NEXT:   arith.addi

// -----

#config1 = #iree_gpu.lowering_config<{lowering_strategy = "print_one"}>
#config2 = #iree_gpu.lowering_config<{lowering_strategy = "print_two"}>
#config3 = #iree_gpu.lowering_config<{lowering_strategy = "print_three"}>
module attributes { transform.with_named_sequence } {
  func.func @multi_config(%arg0: i32) -> i32 {
    %add1 = arith.addi %arg0, %arg0 {lowering_config = #config1} : i32
    %add2 = arith.addi %add1, %add1 {lowering_config = #config2} : i32
    %add3 = arith.addi %add2, %add2 {lowering_config = #config3} : i32
    return %add3 : i32
  }

  transform.named_sequence @print_one(%op: !transform.any_op {transform.readonly}) {
    transform.print %op {name = "one"} : !transform.any_op
    transform.yield
  }
  transform.named_sequence @print_two(%op: !transform.any_op {transform.readonly}) {
    transform.print %op {name = "two"} : !transform.any_op
    transform.yield
  }
  transform.named_sequence @print_three(%op: !transform.any_op {transform.readonly}) {
    transform.print %op {name = "three"} : !transform.any_op
    transform.yield
  }
}

//     CHECK: IR printer:
// CHECK-DAG:   one
// CHECK-DAG:   two
// CHECK-DAG:   three

// -----

#config = #iree_gpu.lowering_config<{lowering_strategy = "lowering_strategy"}>
module {
  func.func @external_strategy(%arg0: i32) -> i32 {
    %add = arith.addi %arg0, %arg0 {lowering_config = #config} : i32
    return %add : i32
  }
}

// See ./external_strategy_spec.mlir for the implementation of
// "lowering_strategy" annotated for this test.
//
// CHECK: IR printer: I am external

// -----

#config = #iree_gpu.lowering_config<{lowering_strategy = "lowering_strategy"}>
module attributes { transform.with_named_sequence } {
  func.func @override_external_strategy(%arg0: i32) -> i32 {
    %add = arith.addi %arg0, %arg0 {lowering_config = #config} : i32
    return %add : i32
  }

  transform.named_sequence @lowering_strategy(%op: !transform.any_op {transform.readonly}) {
    transform.print {name = "I am internal"}
    transform.yield
  }
}

// CHECK: IR printer: I am internal
