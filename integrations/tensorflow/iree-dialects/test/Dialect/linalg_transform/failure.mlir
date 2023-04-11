// RUN: iree-dialects-opt --transform-dialect-interpreter --split-input-file --verify-diagnostics --allow-unregistered-dialect %s

// expected-error @below {{transform dialect interpreter failed}}
module {
  func.func public @no_outlining() {
    // expected-note @below {{target op}}
    "some.operation"() ({}, {}) : () -> ()
    return
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    pdl.pattern @some_operation : benefit(1) {
      %0 = operation "some.operation"
      rewrite %0 with "transform.dialect"
    }

    transform.sequence %arg0: !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
      %0 = pdl_match @some_operation in %arg1 : (!pdl.operation) -> !pdl.operation
      // Make sure we don't crash on wrong operation type.
      // expected-error@below {{failed to outline}}
      transform.loop.outline %0 {func_name = "outlined"} : (!pdl.operation) -> !pdl.operation
    }
  }
}
