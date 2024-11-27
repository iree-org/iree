// RUN: iree-opt --pass-pipeline='builtin.module(builtin.module(iree-codegen-materialize-tuning-specs,iree-codegen-materialize-user-configs))' \
// RUN:   --iree-codegen-tuning-spec-path=%p/tuning_spec.mlir \
// RUN:   --mlir-disable-threading --no-implicit-module %s | FileCheck %s

// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs,builtin.module(iree-codegen-materialize-user-configs))' \
// RUN:   --iree-codegen-tuning-spec-path=%p/tuning_spec.mlir \
// RUN:   --mlir-disable-threading --no-implicit-module %s | FileCheck %s --check-prefix=PARENT

// (1) We start by running the `Materialize Tuning Specs` pass to embed the
// transform dialect library into the module. Doing it by hand hand is not
// possible, because we serialize it as MLIR bytecode.
//
// Check that the transform spec gets executed and that it does not remain as
// a module attribute after `Materialize User Configs`.

// CHECK-LABEL:  [ IR printer: Hello Tuning Spec top-level ]
// CHECK-NEXT:   func.func @main_0
//
// CHECK-LABEL:  module @parent {
// CHECK-LABEL:    module @child {
// CHECK:            func.func @main_0

// (2) Check that the transform spec gets picked up from the **parent** module.
// The tuning spec attribute should remain on the parent module as we
// (conservatively) only remove tuning spec from the module passed
// to the `materialize-user-configs` pass.

// PARENT-LABEL:  [ IR printer: Hello Tuning Spec top-level ]
// PARENT-NEXT:   func.func @main_0
//
// PARENT-LABEL:  module @parent attributes {
// PARENT-SAME:     iree_codegen.tuning_spec_mlirbc = dense<
// PARENT-LABEL:    module @child {
// PARENT:            func.func @main_0

module @parent {
  module @child {
    func.func @main_0() {
      return
    }
  }
}
