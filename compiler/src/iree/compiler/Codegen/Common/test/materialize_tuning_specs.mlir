// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs)' \
// RUN:   --iree-codegen-tuning-spec-path=%p/tuning_spec.mlir \
// RUN:   --iree-codegen-dump-tuning-specs-to=- \
// RUN:   --mlir-disable-threading --no-implicit-module %s | FileCheck %s

// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs)' \
// RUN:   --iree-codegen-tuning-spec-path=%p/tuning_spec_default.mlir \
// RUN:   --iree-codegen-dump-tuning-specs-to=- \
// RUN:   --mlir-disable-threading --no-implicit-module %s | FileCheck %s --check-prefix=SKIPLINK

// Check that the final tuning spec is as expected when the user tuning spec is provided.

// TODO: Add the check for default attribute (`iree_codegen.tuning_spec_with_default_entrypoint`) here
//       once the merging logic supports cases beyond a single `foreach_match` operation.

// CHECK-LABEL: module @iree_linked_tuning_spec
// CHECK-SAME:    transform.with_named_sequence
// CHECK-LABEL:   module @user_spec_0 attributes {transform.with_named_sequence}
// CHECK-LABEL:     transform.named_sequence @hello
// CHECK-SAME:        attributes {iree_codegen.tuning_spec_entrypoint}
// CHECK-LABEL:   transform.named_sequence @__kernel_config
// CHECK:           @user_spec_0::@hello

// Check that the transform spec gets materialized as a module attribute.
// CHECK:        module attributes
// CHECK-SAME:     iree_codegen.tuning_spec_mlirbc = dense<{{.+}}> : vector<{{[0-9]+}}xi8>
// CHECK-LABEL:    func.func @main_0


// CHECK that the user-provided tuning spec is materized without linking when default tuing spec
// is missing and the user-provided tuning spec is marked the default attribute.

// SKIPLINK-LABEL: module  @user_spec
// SKIPLINK-SAME:    iree_codegen.tuning_spec_with_default_entrypoint
// SKIPLINK-SAME:    transform.with_named_sequence
// SKIPLINK:         transform.print  {name = "Hello Tuning Spec"}
// SKIPLINK-NOT:    module @{{.+}}
// SKIPLINK:        module attributes
// SKIPLINK-SAME:     iree_codegen.tuning_spec_mlirbc = dense<{{.+}}> : vector<{{[0-9]+}}xi8>
// SKIPLINK-LABEL:    func.func @main_0
module {
  func.func @main_0() {
    return
  }
}
