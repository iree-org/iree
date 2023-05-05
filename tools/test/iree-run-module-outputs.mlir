// Tests that execution providing no outputs is ok.

// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-sync --module=- --function=no_output) | \
// RUN: FileCheck --check-prefix=NO-OUTPUT %s
// NO-OUTPUT-LABEL: EXEC @no_output
func.func @no_output() {
  return
}

// -----

// Tests the default output printing to stdout.

// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-sync --module=- --function=default) | \
// RUN: FileCheck --check-prefix=OUTPUT-DEFAULT %s
// OUTPUT-DEFAULT-LABEL: EXEC @default
func.func @default() -> (i32, tensor<f32>, tensor<?x4xi32>) {
  // OUTPUT-DEFAULT: result[0]: i32=123
  %0 = arith.constant 123 : i32
  // OUTPUT-DEFAULT: result[1]: hal.buffer_view
  // OUTPUT-DEFAULT-NEXT: f32=4
  %1 = arith.constant dense<4.0> : tensor<f32>
  // OUTPUT-DEFAULT: result[2]: hal.buffer_view
  // OUTPUT-DEFAULT-NEXT: 2x4xi32=[0 1 2 3][4 5 6 7]
  %2 = flow.tensor.constant dense<[[0,1,2,3],[4,5,6,7]]> : tensor<2x4xi32> -> tensor<?x4xi32>
  return %0, %1, %2 : i32, tensor<f32>, tensor<?x4xi32>
}

// -----

// Tests explicit output to npy files by producing a concatenated .npy and then
// printing the results in python. This also verifies our npy files can be
// parsed by numpy.

// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-sync --module=- --function=numpy \
// RUN:                  --output= \
// RUN:                  --output=@%t \
// RUN:                  --output=+%t) && \
// RUN:  python3 %S/echo_npy.py %t | \
// RUN: FileCheck --check-prefix=OUTPUT-NUMPY %s
func.func @numpy() -> (i32, tensor<f32>, tensor<?x4xi32>) {
  // Output skipped:
  %0 = arith.constant 123 : i32
  // OUTPUT-NUMPY{LITERAL}: 4.0
  %1 = arith.constant dense<4.0> : tensor<f32>
  // OUTPUT-NUMPY-NEXT{LITERAL}: [[0 1 2 3]
  // OUTPUT-NUMPY-NEXT{LITERAL}:  [4 5 6 7]]
  %2 = flow.tensor.constant dense<[[0,1,2,3],[4,5,6,7]]> : tensor<2x4xi32> -> tensor<?x4xi32>
  return %0, %1, %2 : i32, tensor<f32>, tensor<?x4xi32>
}
