// RUN: iree-opt --pass-pipeline='builtin.module(iree-codegen-materialize-tuning-specs)' \
// RUN:   --iree-codegen-tuning-spec-path=%s --no-implicit-module --verify-diagnostics %s

// Check that we error out on mlir inputs that are not tuning specs (e.g., the input itself).

// expected-error@+2 {{Module without the 'transform.with_named_sequence' attribute is not a transform dialect library}}
// expected-error@+1 {{Failed to load tuning spec transform dialect library from}}
module {
  func.func @main_0() {
    return
  }
}
