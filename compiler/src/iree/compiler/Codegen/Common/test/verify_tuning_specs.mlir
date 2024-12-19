// RUN: iree-opt --verify-diagnostics --split-input-file %s

module @foo_module attributes { transform.with_named_sequence } {
  func.func @baz(%arg0: i32) -> () {
    return
  }
  transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.something } {
    transform.yield %arg0 : !transform.any_op
  }
  // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
    transform.yield %arg0 : !transform.any_op
  }
}

// -----

module @foo_module attributes { transform.with_named_sequence } {
  // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.yield %arg0 : !transform.any_op
  }
}

// -----

module @foo_module attributes { transform.with_named_sequence } {
  // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
  transform.named_sequence @foo(%arg0: i32) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {}
}

// -----

module @foo_module attributes { transform.with_named_sequence } {
  // expected-error @+1{{Tuning spec entry point expected to return any_op}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> i32
    attributes { iree_codegen.tuning_spec_entrypoint } {
    %0 = arith.constant 0 : i32
    transform.yield %0 : i32
  }
}

// -----

module @foo_module attributes { transform.with_named_sequence } {
  // expected-error @+1{{Tuning spec entry point expected to return any_op}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly})
    attributes { iree_codegen.tuning_spec_entrypoint } {}
}

// -----

// expected-error @+1{{The tuning specification must include a named sequence with the symbol name '__kernel_config'}}
module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
}

// -----

// expected-error @+1{{The tuning specification must include a named sequence with the symbol name '__kernel_config'}}
module @iree_default_tuning_spec attributes { iree_codegen.tuning_spec_with_default_entrypoint } {
  func.func @__kernel_config(%arg0: i32) -> () {
    return
  }
}
