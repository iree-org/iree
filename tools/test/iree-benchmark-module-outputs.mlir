// Tests that output processing is performed by iree-benchmark-module when
// --enable_output_processing is passed (and only if it is passed).
// Test that --output= options are working in general, by testing a single one.
// See iree-run-module-outputs.mlir for tests of the --output= options
// themselves, which are implemented by code shared between iree-run-module
// and iree-benchmark-module.

// RUN: (iree-compile --iree-hal-target-device=local \
// RUN:               --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-benchmark-module --device=local-sync --module=- \
// RUN:                        --function=default) | \
// RUN: FileCheck --check-prefix=DISABLED %s
// DISABLED-LABEL: BM_default
// DISABLED-NOT: result[

// RUN: (iree-compile --iree-hal-target-device=local \
// RUN:               --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-benchmark-module --device=local-sync --module=- \
// RUN:                        --function=default \
// RUN:                        --enable_output_processing) | \
// RUN: FileCheck --check-prefix=ENABLED %s
// ENABLED-LABEL: BM_default
// ENABLED: result[0]: i32=123

// RUN: (iree-compile --iree-hal-target-device=local \
// RUN:               --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-benchmark-module --device=local-sync --module=- \
// RUN:                        --function=default \
// RUN:                        --enable_output_processing \
// RUN:                        --output=@%t.npy) && \
// RUN:  "%PYTHON" %S/echo_npy.py %t.npy | \
// RUN: FileCheck --check-prefix=OUTPUT-OPTION %s
// OUTPUT-OPTION{LITERAL}: 123
func.func @default() -> (i32) {
  %0 = arith.constant 123 : i32
  return %0 : i32
}

// Tests that a failing --expected_output= check makes the tool exit with a
// failure exit code, same as iree-run-module. See
// iree-run-module-expected.mlir for full coverage of --expected_output=
// comparison behavior itself.

// RUN: (iree-compile --iree-hal-target-device=local \
// RUN:               --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  not iree-benchmark-module --device=local-sync --module=- \
// RUN:                            --function=abs --input=f32=-2 \
// RUN:                            --enable_output_processing \
// RUN:                            --expected_output=f32=3) | \
// RUN: FileCheck --check-prefix=EXPECTED-MISMATCH %s
// EXPECTED-MISMATCH-LABEL: BM_abs
// EXPECTED-MISMATCH: [FAILED]
func.func @abs(%input: tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}
