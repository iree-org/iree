// RUN: iree-opt --no-implicit-module --verify-diagnostics -split-input-file --mlir-disable-threading %s

module @td_module attributes { transform.with_named_sequence } {
  module @foo_module attributes { transform.with_named_sequence } {
    // expected-error @+1{{'iree_codegen.tuning_spec_entrypoint' attribute must be a UnitAttr}}
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint = "foo" } {
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
    func.func @baz(%arg0: i32) -> () {
      return
    }
  }
}
