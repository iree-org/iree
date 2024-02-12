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
// RUN:                  --output=@%t.npy \
// RUN:                  --output=+%t.npy) && \
// RUN:  "%PYTHON" %S/echo_npy.py %t.npy | \
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

// -----

// Tests output to binary files by round-tripping the output of a function into
// another invocation reading from the binary files. Each output is written to
// its own file (optimal for alignment/easier to inspect).

// RUN: (iree-compile --iree-hal-target-backends=vmvx %s -o=%t.vmfb && \
// RUN:  iree-run-module --device=local-sync \
// RUN:                  --module=%t.vmfb \
// RUN:                  --function=write_binary \
// RUN:                  --output=@%t.0.bin \
// RUN:                  --output=@%t.1.bin && \
// RUN:  iree-run-module --device=local-sync \
// RUN:                  --module=%t.vmfb \
// RUN:                  --function=echo_binary \
// RUN:                  --input=f32=@%t.0.bin \
// RUN:                  --input=2x4xi32=@%t.1.bin) | \
// RUN: FileCheck --check-prefix=OUTPUT-BINARY %s

// Tests output to binary files by round-tripping the output of a function into
// another invocation reading from the binary files. The values are appended to
// a single file and read from the single file.

// RUN: (iree-compile --iree-hal-target-backends=vmvx %s -o=%t.vmfb && \
// RUN:  iree-run-module --device=local-sync \
// RUN:                  --module=%t.vmfb \
// RUN:                  --function=write_binary \
// RUN:                  --output=@%t.bin \
// RUN:                  --output=+%t.bin && \
// RUN:  iree-run-module --device=local-sync \
// RUN:                  --module=%t.vmfb \
// RUN:                  --function=echo_binary \
// RUN:                  --input=f32=@%t.bin \
// RUN:                  --input=2x4xi32=+%t.bin) | \
// RUN: FileCheck --check-prefix=OUTPUT-BINARY %s

func.func @write_binary() -> (tensor<f32>, tensor<?x4xi32>) {
  %0 = arith.constant dense<4.0> : tensor<f32>
  %1 = flow.tensor.constant dense<[[0,1,2,3],[4,5,6,7]]> : tensor<2x4xi32> -> tensor<?x4xi32>
  return %0, %1 : tensor<f32>, tensor<?x4xi32>
}
func.func @echo_binary(%arg0: tensor<f32>, %arg1: tensor<?x4xi32>) -> (tensor<f32>, tensor<?x4xi32>) {
  // OUTPUT-BINARY{LITERAL}: f32=4
  // OUTPUT-BINARY{LITERAL}: 2x4xi32=[0 1 2 3][4 5 6 7]
  return %arg0, %arg1 : tensor<f32>, tensor<?x4xi32>
}
