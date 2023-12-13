// Passing no inputs is okay.

// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-sync --module=- --function=no_input) | \
// RUN: FileCheck --check-prefix=NO-INPUT %s
// NO-INPUT-LABEL: EXEC @no_input
func.func @no_input() {
  return
}

// -----

// Scalars use the form `--input=value`. Type (float/int) should be omitted.
//   * Type conversions between bit depths (e.g. i8 -> i32) may occur!

// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-sync --module=- --function=scalars \
// RUN:  --input=1 --input=5 --input=1234 --input=-3.14) | \
// RUN: FileCheck --check-prefix=INPUT-SCALARS %s
// INPUT-SCALARS-LABEL: EXEC @scalars
func.func @scalars(%arg0: i1, %arg1: i8, %arg2 : i32, %arg3 : f32) -> (i1, i8, i32, f32) {
  // INPUT-SCALARS: result[0]: i32=1
  // INPUT-SCALARS: result[1]: i32=5
  // INPUT-SCALARS: result[2]: i32=1234
  // INPUT-SCALARS: result[3]: f32=-3.14
  return %arg0, %arg1, %arg2, %arg3 : i1, i8, i32, f32
}

// -----

// Buffers ("tensors") use the form `--input=[shape]xtype=[value]`.
//   * If any values are omitted, zeroes will be used.
//   * Quotes should be used around values with spaces.
//   * Brackets may also be used to separate element values.

// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-sync --module=- --function=buffers \
// RUN:  --input=i32=5 --input=2xi32 --input="2x3xi32=1 2 3 4 5 6") | \
// RUN: FileCheck --check-prefix=INPUT-BUFFERS %s
// INPUT-BUFFERS-LABEL: EXEC @buffers
func.func @buffers(%arg0: tensor<i32>, %arg1: tensor<2xi32>, %arg2: tensor<2x3xi32>) -> (tensor<i32>, tensor<2xi32>, tensor<2x3xi32>) {
  // INPUT-BUFFERS: result[0]: hal.buffer_view
  // INPUT-BUFFERS-NEXT: i32=5
  // INPUT-BUFFERS: result[1]: hal.buffer_view
  // INPUT-BUFFERS-NEXT: 2xi32=0 0
  // INPUT-BUFFERS: result[2]: hal.buffer_view
  // INPUT-BUFFERS-NEXT: 2x3xi32=[1 2 3][4 5 6]
  return %arg0, %arg1, %arg2 : tensor<i32>, tensor<2xi32>, tensor<2x3xi32>
}

// -----

// Buffer values can be read from binary files with `@some/file.bin`.
//   * numpy npy files from numpy.save or previous tooling output can be read to
//     provide 1+ values.
//   * Some data types may be converted (i32 -> si32 here)

// RUN: iree-compile --iree-hal-target-backends=vmvx %s -o %t.vmfb
//
// RUN: iree-run-module --device=local-sync --module=%t.vmfb --function=npy_round_trip \
// RUN:   --input="2xi32=11 12" --input="3xi32=1 2 3" --output=@%t.npy --output=+%t.npy
//
// RUN: iree-run-module --device=local-sync --module=%t.vmfb --function=npy_round_trip \
// RUN:   --input=@%t.npy | \
// RUN: FileCheck --check-prefix=INPUT-NPY_FILES %s

// INPUT-NPY_FILES-LABEL: EXEC @npy_round_trip
func.func @npy_round_trip(%arg0: tensor<2xi32>, %arg1: tensor<3xi32>) -> (tensor<2xi32>, tensor<3xi32>) {
  // INPUT-NPY_FILES: result[0]: hal.buffer_view
  // INPUT-NPY_FILES-NEXT: 2xsi32=11 12
  // INPUT-NPY_FILES: result[1]: hal.buffer_view
  // INPUT-NPY_FILES-NEXT: 3xsi32=1 2 3
  return %arg0, %arg1 : tensor<2xi32>, tensor<3xi32>
}
