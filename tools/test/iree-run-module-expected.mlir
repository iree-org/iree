// RUN: (iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=-2 --expected_output=f32=2.0) | \
// RUN:  FileCheck %s --check-prefix=SUCCESS-MATCHES
// RUN: (iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=-2 --expected_output="(ignored)") | \
// RUN:  FileCheck %s --check-prefix=SUCCESS-IGNORED
// RUN: (iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=-2 --expected_output=f32=2.1 --expected_f32_threshold=0.1) | \
// RUN:  FileCheck %s --check-prefix=SUCCESS-THRESHOLD
// RUN: (iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  not iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=123 --expected_output=f32=2.0) | \
// RUN:  FileCheck %s --check-prefix=FAILED-FIRST
// RUN: (iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  not iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=-2 --expected_output=f32=4.5) | \
// RUN:  FileCheck %s --check-prefix=FAILED-SECOND
// RUN: (iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  not iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=-2 --expected_output=4xf32=2.0) | \
// RUN:  FileCheck %s --check-prefix=FAILED-SHAPE
// RUN: iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx %s -o %t.vmfb
// RUN: iree-run-module --device=local-task --module=%t.vmfb --function=identity_i64 --input=i64=8 --output=@%t.npy
// RUN: iree-run-module --device=local-task --module=%t.vmfb --function=identity_i64 --input=i64=8 --expected_output=i64=@%t.npy | \
// RUN:  FileCheck %s --check-prefix=SUCCESS-NPY-SIGNLESS-INTEGER
// RUN: (not iree-run-module --device=local-task --module=%t.vmfb --function=identity_i64 --input=i64=8 --expected_output=si64=8) | \
// RUN:  FileCheck %s --check-prefix=FAILED-EXPLICIT-SIGNED-INTEGER
// RUN: not iree-run-module --device=local-task --module=%t.vmfb --function=identity_i64 --input=i64=8 --expected_output=1xi64=@%t.npy 2>&1 | \
// RUN:  FileCheck %s --check-prefix=FAILED-NPY-SHAPE
// RUN: not iree-run-module --device=local-task --module=%t.vmfb --function=identity_i64 --input=i64=8 --expected_output=f64=@%t.npy 2>&1 | \
// RUN:  FileCheck %s --check-prefix=FAILED-NPY-ELEMENT-TYPE

// SUCCESS-MATCHES: [SUCCESS]
// SUCCESS-THRESHOLD: [SUCCESS]
// SUCCESS-IGNORED: [SUCCESS]
// FAILED-FIRST: [FAILED] result[0]: element at index 0 (-2) does not match the expected (123)
// FAILED-SECOND: [FAILED] result[1]: element at index 0 (2) does not match the expected (4.5)
// FAILED-SHAPE: [FAILED] result[1]: metadata is f32; expected that the view matches 4xf32
// SUCCESS-NPY-SIGNLESS-INTEGER: [SUCCESS]
// FAILED-EXPLICIT-SIGNED-INTEGER: [FAILED] result[0]: metadata is i64; expected that the view matches si64
// FAILED-NPY-SHAPE: explicit shape rank 1 does not match NPY shape rank 0
// FAILED-NPY-ELEMENT-TYPE: explicit element type is incompatible with NPY element type

func.func @abs(%input: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %input, %result : tensor<f32>, tensor<f32>
}

func.func @identity_i64(%input: tensor<i64>) -> tensor<i64> {
  return %input : tensor<i64>
}
