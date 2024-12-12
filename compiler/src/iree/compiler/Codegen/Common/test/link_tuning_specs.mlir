// RUN: iree-opt %s --no-implicit-module --iree-codegen-link-tuning-specs --split-input-file \
// RUN:   | FileCheck %s

// CHECK-LABEL: module @td_module_0
//
// CHECK:         transform.named_sequence @outer_spec
//
// CHECK:         transform.named_sequence @__kernel_config
// CHECK-SAME:      (%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
// CHECK-SAME:      attributes {iree_codegen.tuning_spec_entrypoint}
// CHECK:           %[[OP1:.+]] = transform.include @foo_module::@foo failures(suppress) (%arg0)
// CHECK-NEXT:      %[[OP2:.+]] = transform.include @bar_module::@bar failures(suppress) (%[[OP1]])
// CHECK-NEXT:      %[[OP3:.+]] = transform.include @baz_module::@baz failures(suppress) (%[[OP2]])
// CHECK-NEXT:      transform.yield %[[OP3]] : !transform.any_op

module @td_module_0 attributes { transform.with_named_sequence } {
  module @foo_module attributes { transform.with_named_sequence } {
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.print {name = "Foo", skip_regions}
      transform.yield %arg0 : !transform.any_op
    }
  }

  module @bar_module attributes { transform.with_named_sequence } {
    transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
      transform.print {name = "Bar", skip_regions}
      transform.yield %arg0 : !transform.any_op
    }
  }

  module @baz_module attributes { transform.with_named_sequence } {
    transform.named_sequence @baz(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.print {name = "Baz", skip_regions}
      transform.yield %arg0 : !transform.any_op
    }
  }

  transform.named_sequence @outer_spec(%module: !transform.any_op {transform.readonly}) -> !transform.any_op
    attributes { iree_codegen.tuning_spec_entrypoint } {
    transform.yield %module : !transform.any_op
  }
}


// -----

// Here, `foo` shouldn't be included because it's not marked with `tuning_spec_entrypoint`.

// CHECK-LABEL: module @td_module_1
// CHECK:       @foo_module
// CHECK:       @__kernel_config(
// CHECK-NOT      transform.include @foo_module::@foo failures(suppress) (%arg0) : (!transform.any_op) -> !transform.any_op
// CHECK:         transform.include @foo_module::@bar failures(suppress) (%arg0) : (!transform.any_op) -> !transform.any_op
// CHECK-NEXT:    transform.yield

module @td_module_1 attributes { transform.with_named_sequence } {
  module @foo_module attributes { transform.with_named_sequence } {
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
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


// -----

// Make sure we do not crash on modules with no tuning specs.

// CHECK-LABEL: module @td_module_2
// CHECK-NOT:   @__kernel_config
module @td_module_2 attributes { transform.with_named_sequence } {}

// -----

// Make sure we do not crash on unnamed nested modules.

// CHECK-LABEL: module @td_module_3
// CHECK:       transform.named_sequence @foo
// CHECK-NOT:     @__kernel_config

module @td_module_3 attributes { transform.with_named_sequence } {
  module attributes { transform.with_named_sequence } {
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
  }
}

// -----

// Make sure that the names of all included specs and the outermost entrypoint
// are kept unique.

// CHECK-LABEL: module @td_module_4
// CHECK:       @foo_module attributes
// CHECK:       @bar_module attributes
// CHECK:       @__kernel_config(
// CHECK:         transform.include @foo_module::@foo failures(suppress) (%arg0) : (!transform.any_op) -> !transform.any_op
// CHECK:         transform.include @foo_module::@__kernel_config_1 failures(suppress)
// CHECK:         transform.include @bar_module::@foo_1 failures(suppress)
// CHECK:         transform.include @bar_module::@__kernel_config_2 failures(suppress)
// CHECK-NEXT:    transform.yield

module @td_module_4 attributes { transform.with_named_sequence } {
  module @foo_module attributes { transform.with_named_sequence } {
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
  }
  module @bar_module attributes { transform.with_named_sequence } {
    transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @__kernel_config(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
      attributes { iree_codegen.tuning_spec_entrypoint } {
      transform.yield %arg0 : !transform.any_op
    }
  }
}
