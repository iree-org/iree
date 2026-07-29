// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(torch-iree-sync-func-conversion)" --verify-diagnostics %s

// Externalized transients require the coarse-fences ABI; the flag combination
// is rejected at plugin activation before any compilation runs.
// RUN: not iree-compile --iree-torch-emit-async-entry-points=false --iree-torch-externalize-transients %s 2>&1 | FileCheck --check-prefix=CHECK-CONFLICT %s
// CHECK-CONFLICT: iree-torch-externalize-transients requires async entry points

// Mutable tensors require the coarse-fences ABI for in-place aliasing and are
// not supported by the sync-only conversion.
builtin.module @mutable_arg {
// expected-error @+1 {{mutable tensors are not supported}}
func.func @main(%arg0: !torch.tensor<[5,4],f32>) {
  return
}
}

// -----
builtin.module @mutable_result {
// expected-error @+1 {{mutable tensors are not supported}}
func.func @main() -> !torch.tensor<[5,4],f32> {
  %0 = torch.operator "some.mutable_producer"() : () -> !torch.tensor<[5,4],f32>
  return %0 : !torch.tensor<[5,4],f32>
}
}
