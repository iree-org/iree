// RUN: iree-compile %s -o ignored.mlir \
// RUN:     --iree-hal-target-backends=vmvx \
// RUN:     --iree-hal-dump-executable-benchmarks-to=- | \
// RUN: iree-compile - | \
// RUN: iree-benchmark-module --module=- | \
// RUN: FileCheck %s

// This test relies on us piping stdout and that there's only a single
// executable (otherwise we'd need to look at files and that's harder
// cross-platform). Real automation of this requires xargs: compile and dump a
// directory of .mlir benchmarks by specifying a path to the dump flag instead
// of `-` (indicating stdout) and then ls | xargs them to iree-compile to
// produce the vmfbs (optionally with different flags for each) and run
// iree-benchmark-module to run them.
//
// Example of dumping per-dispatch executable benchmarks, compiling each, and
// then benchmarking each:
//  iree-compile some_input.mlir -o ignored.mlir \
//      --iree-hal-target-backends=vmvx \
//      --iree-hal-dump-executable-benchmarks-to=benchmarks/ | \
//  ls -1 benchmarks/ | xargs -i sh -c "iree-compile benchmarks/{} | iree-benchmark-module --module=-"
//
// NOTE: only dispatches that are able to be benchmarked automatically will be
// written; if you don't end up with a .mlir file for the dispatch you're
// interested in then either the pass needs improvement or the usage needs to be
// reduced/simplified. Dynamic shapes, for example, will usually stop a dispatch
// from being benchmarkable without explicit shape arguments.

// CHECK: BM_abs_dispatch_0_vmvx_bytecode_fb_abs_dispatch_0_generic
func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}
