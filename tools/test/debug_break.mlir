// Tests the --iree-debug-break-{before,after} instrumentation in auto-patch
// mode. The patch file replaces the module body at the chosen pass boundary;
// the compile then resumes and emits IR reflecting the patch.
//
// This test proves:
//   * The break fires at the requested pass (BEGIN/END banner on stderr).
//   * The patch file is actually applied (patched symbol appears in output).
//   * Both --iree-debug-break-before and --iree-debug-break-after work.

// Break-after: patch is applied AFTER iree-sanitize-module-names, so subsequent
// passes (ABI wrapping, etc.) operate on the patched IR.
// RUN: iree-compile %s \
// RUN:     --iree-hal-target-device=local \
// RUN:     --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:     --compile-to=abi \
// RUN:     --iree-debug-break-after=iree-sanitize-module-names \
// RUN:     --iree-debug-break-file=%t-after.mlir \
// RUN:     --iree-debug-break-test-patch-file=%p/debug_break_patch.mlir \
// RUN:     2> %t-after.stderr \
// RUN:   | FileCheck %s --check-prefix=OUT
// RUN: FileCheck --check-prefix=BANNER-AFTER %s < %t-after.stderr

// BANNER-AFTER: [iree-debug-break] BEGIN phase=after pass=iree-sanitize-module-names
// BANNER-AFTER: [iree-debug-break] auto-patching from
// BANNER-AFTER: [iree-debug-break] END phase=after pass=iree-sanitize-module-names

// Break-before: same patch file, different hook. Effect on final IR is the
// same here because the patch-file's module overwrites the live one entirely.
// RUN: iree-compile %s \
// RUN:     --iree-hal-target-device=local \
// RUN:     --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:     --compile-to=abi \
// RUN:     --iree-debug-break-before=iree-abi-wrap-entry-points \
// RUN:     --iree-debug-break-file=%t-before.mlir \
// RUN:     --iree-debug-break-test-patch-file=%p/debug_break_patch.mlir \
// RUN:     2> %t-before.stderr \
// RUN:   | FileCheck %s --check-prefix=OUT
// RUN: FileCheck --check-prefix=BANNER-BEFORE %s < %t-before.stderr

// BANNER-BEFORE: [iree-debug-break] BEGIN phase=before pass=iree-abi-wrap-entry-points
// BANNER-BEFORE: [iree-debug-break] auto-patching from
// BANNER-BEFORE: [iree-debug-break] END phase=before pass=iree-abi-wrap-entry-points

// Stdin-mode round-trip: feed "continue" on stdin, let the break fire
// and resume without any edit. This catches banner-format drift that would
// hang stdin-driven workflows.
// RUN: echo continue | iree-compile %s \
// RUN:     --iree-hal-target-device=local \
// RUN:     --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:     --compile-to=abi \
// RUN:     --iree-debug-break-after=iree-sanitize-module-names \
// RUN:     --iree-debug-break-file=%t-stdin.mlir \
// RUN:     2> %t-stdin.stderr \
// RUN:   | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: FileCheck --check-prefix=BANNER-STDIN %s < %t-stdin.stderr

// BANNER-STDIN: [iree-debug-break] BEGIN phase=after pass=iree-sanitize-module-names
// BANNER-STDIN: Edit the file, then type 'continue'
// BANNER-STDIN: [iree-debug-break] END phase=after pass=iree-sanitize-module-names

// File-mode round-trip: pre-create the .continue sentinel before
// launching so the compiler observes it on the first poll tick and resumes
// without needing a concurrent driver.
// RUN: rm -f %t-file.mlir.continue %t-file.mlir.abort && \
// RUN:   touch %t-file.mlir.continue && \
// RUN:   iree-compile %s \
// RUN:     --iree-hal-target-device=local \
// RUN:     --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:     --compile-to=abi \
// RUN:     --iree-debug-break-after=iree-sanitize-module-names \
// RUN:     --iree-debug-break-file=%t-file.mlir \
// RUN:     --iree-debug-break-mode=file \
// RUN:     2> %t-file.stderr \
// RUN:   | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: FileCheck --check-prefix=BANNER-FILE-MODE %s < %t-file.stderr

// BANNER-FILE-MODE: [iree-debug-break] BEGIN phase=after pass=iree-sanitize-module-names
// BANNER-FILE-MODE: touch {{.*}}.continue{{.*}} to signal
// BANNER-FILE-MODE: [iree-debug-break] END phase=after pass=iree-sanitize-module-names

func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// The patched module renames the entry point to @patched_add and switches
// addf -> mulf, so the final ABI-wrapped IR reflects both changes.
// OUT: util.func public @patched_add
// OUT-SAME: !hal.buffer_view
// OUT: hal.tensor.import
// OUT: arith.mulf
// OUT-NOT: arith.addf
// OUT: hal.tensor.export

// Round-trip runs (stdin and file mode) make no edits, so the unwrapped
// @add function survives through ABI wrapping unchanged.
// ROUNDTRIP: util.func public @add
// ROUNDTRIP-SAME: !hal.buffer_view
// ROUNDTRIP: hal.tensor.import
// ROUNDTRIP: arith.addf
// ROUNDTRIP: hal.tensor.export
