// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs)' \
// RUN:   --iree-codegen-tuning-spec-path=%p/tuning_spec.mlir \
// RUN:   --iree-codegen-dump-tuning-specs-to=- \
// RUN:   --mlir-disable-threading --no-implicit-module %s | FileCheck %s

// Check that the final tuning spec is as expected when the user tuning spec is provided.

// CHECK-LABEL: module @iree_linked_tuning_spec attributes {transform.with_named_sequence}
// CHECK-LABEL:   module @user_spec_0 attributes {transform.with_named_sequence}
// CHECK-LABEL:     transform.named_sequence @hello
// CHECK-SAME:        attributes {iree_codegen.tuning_spec_entrypoint}
// CHECK-LABEL:   transform.named_sequence @__kernel_config
// CHECK:           @user_spec_0::@hello

// Check that the transform spec gets materialized as a module attribute.
// CHECK:        module attributes
// CHECK-SAME:     iree_codegen.tuning_spec_mlirbc = dense<{{.+}}> : vector<{{[0-9]+}}xi8>
// CHECK-LABEL:    func.func @main_0

module {
  func.func @main_0() {
    return
  }
}
