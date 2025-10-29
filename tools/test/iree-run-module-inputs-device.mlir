
// Verify parsing of signless small integer types passes either signed or
// unsigned range checks.

// RUN: (iree-compile --iree-hal-target-device=local \
// RUN:               --iree-hal-local-target-device-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-sync \
// RUN:                  --module=- \
// RUN:                  --function=device_assignment \
// RUN:                  --input="2xi16=65535 -32767" \
// RUN:                  --input="3xi8=-6 250 0xFF") | \
// RUN: FileCheck --check-prefix=DEVICE-ASSIGNMENT %s
// DEVICE-ASSIGNMENT-LABEL: EXEC @device_assignment
func.func @device_assignment(%arg0: tensor<2xi16>, %arg1: tensor<3xi8>) -> (tensor<2xi16>, tensor<3xi8>) {
  // Signedness of printing signless values is unspecified.
  // DEVICE-ASSIGNMENT: result[0]: hal.buffer_view
  // DEVICE-ASSIGNMENT-NEXT: 2xi16=-2 2
  // DEVICE-ASSIGNMENT: result[1]: hal.buffer_view
  // DEVICE-ASSIGNMENT-NEXT: 3xi8=-12 -12 -2
  %0 = arith.addi %arg0, %arg0 : tensor<2xi16>
  %1 = arith.addi %arg1, %arg1 : tensor<3xi8>
  return %0, %1 : tensor<2xi16>, tensor<3xi8>
}

