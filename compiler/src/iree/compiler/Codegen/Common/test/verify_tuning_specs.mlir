// RUN: iree-opt  --verify-diagnostics --split-input-file  %s

module @td_module attributes { transform.with_named_sequence } {
  module @foo_module attributes { transform.with_named_sequence } {
    // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.something } {
      transform.yield %arg0 : !transform.any_op
    }
    func.func @baz(%arg0: i32) -> () {
      return
    }
  }
}

// -----

module @td_module attributes { transform.with_named_sequence } {
  module @foo_module attributes { transform.with_named_sequence } {
    // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}, %arg1: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
    func.func @baz(%arg0: i32) -> () {
      return
    }
  }
}

// -----

module @td_module attributes { transform.with_named_sequence } {
  module @foo_module attributes { transform.with_named_sequence } {
    // expected-error @+1{{Tuning spec entry point expected to return any_op}}
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> i32
      attributes { iree_codegen.tuning_spec_entrypoint } {
      %0 = arith.constant 0 : i32
      transform.yield %0 : i32
    }
    func.func @baz(%arg0: i32) -> () {
      return
    }
  }
}

// -----

module @td_module attributes { transform.with_named_sequence } {
  module @foo_module attributes { transform.with_named_sequence } {
    // expected-error @+1{{Tuning spec entry point expected to have a single any_op argument}}
    transform.named_sequence @foo(%arg0: i32) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {

    }
    func.func @baz(%arg0: i32) -> () {
      return
    }
  }
}
