// RUN: iree-opt %s --split-input-file --iree-transform-dialect-interpreter --verify-diagnostics 

module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{match registry not available}}
    transform.iree.match_callback failures(suppress) "_test_match_callback"() : () -> ()
  }
}

// -----

module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.iree.register_match_callbacks
    // expected-error @below {{callback '_non_existing_name_' not found in the registry}}
    transform.iree.match_callback failures(suppress) "_non_existing_name_"() : () -> ()
  }
}

// -----

module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.iree.register_match_callbacks
    // expected-error @below {{callback produced a different number of handles than expected}}
    transform.iree.match_callback failures(suppress) "_test_match_callback"(%arg0) : (!transform.any_op) -> ()
  }
}

// -----

// Successful match.
module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.iree.register_match_callbacks
    transform.iree.match_callback failures(propagate) "_test_match_callback"(%arg0) : (!transform.any_op) -> (!transform.any_op)
  }
}

// -----

module attributes {test.iree_transform_do_not_match} {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.iree.register_match_callbacks
    // expected-error @below {{failed to match}}
    transform.iree.match_callback failures(propagate) "_test_match_callback"(%arg0) : (!transform.any_op) -> (!transform.any_op)
  }
}

// -----

// Failed to match, but the op silences such errors.
module attributes {test.iree_transform_do_not_match} {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.iree.register_match_callbacks
    transform.iree.match_callback failures(suppress) "_test_match_callback"(%arg0) : (!transform.any_op) -> (!transform.any_op)
  }
}

// -----

// Failed to match, but the parent sequence silences all errors.
module attributes {test.iree_transform_do_not_match} {
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    transform.iree.register_match_callbacks
    transform.iree.match_callback failures(propagate) "_test_match_callback"(%arg0) : (!transform.any_op) -> (!transform.any_op)
  }
}
