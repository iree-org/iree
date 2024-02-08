// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=-2 --expected_output=f32=2.0) | FileCheck %s --check-prefix=SUCCESS-MATCHES
// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=-2 --expected_output="(ignored)") | FileCheck %s --check-prefix=SUCCESS-IGNORED
// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=-2 --expected_output=f32=2.1 --expected_f32_threshold=0.1) | FileCheck %s --check-prefix=SUCCESS-THRESHOLD
// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | not iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=123 --expected_output=f32=2.0) | FileCheck %s --check-prefix=FAILED-FIRST
// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | not iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=-2 --expected_output=f32=4.5) | FileCheck %s --check-prefix=FAILED-SECOND
// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | not iree-run-module --device=local-task --module=- --function=abs --input=f32=-2 --expected_output=f32=-2 --expected_output=4xf32=2.0) | FileCheck %s --check-prefix=FAILED-SHAPE

// SUCCESS-MATCHES: [SUCCESS]
// SUCCESS-THRESHOLD: [SUCCESS]
// SUCCESS-IGNORED: [SUCCESS]
// FAILED-FIRST: [FAILED] result[0]: element at index 0 (-2) does not match the expected (123)
// FAILED-SECOND: [FAILED] result[1]: element at index 0 (2) does not match the expected (4.5)
// FAILED-SHAPE: [FAILED] result[1]: metadata is f32; expected that the view matches 4xf32

func.func @abs(%input: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %input, %result : tensor<f32>, tensor<f32>
}
