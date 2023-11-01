// RUN: iree-opt %s --split-input-file --iree-transform-dialect-interpreter --verify-diagnostics 

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    // expected-error @below {{match registry not available}}
    transform.iree.match_callback failures(suppress) "_test_match_callback"() : () -> ()
    transform.yield
  } // @__transform_main
} // module


// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks
    // expected-error @below {{callback '_non_existing_name_' not found in the registry}}
    transform.iree.match_callback failures(suppress) "_non_existing_name_"() : () -> ()
    transform.yield
  } // @__transform_main
} // module


// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks
    // expected-error @below {{callback produced a different number of handles than expected}}
    transform.iree.match_callback failures(suppress) "_test_match_callback"(%root) : (!transform.any_op) -> ()
    transform.yield
  } // @__transform_main
} // module


// -----

// Successful match.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks
    transform.iree.match_callback failures(propagate) "_test_match_callback"(%root) : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  } // @__transform_main
} // module


// -----

module attributes { transform.with_named_sequence , test.iree_transform_do_not_match } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks
    // expected-error @below {{failed to match}}
    transform.iree.match_callback failures(propagate) "_test_match_callback"(%root) : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  } // @__transform_main
} // module


// -----

// Failed to match, but the op silences such errors.
module attributes { transform.with_named_sequence, test.iree_transform_do_not_match } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks
    transform.iree.match_callback failures(suppress) "_test_match_callback"(%root) : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  } // @__transform_main
} // module

