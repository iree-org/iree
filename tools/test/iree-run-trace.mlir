// Tests iree-run-trace usage by running two calls of @mul and passing the
// result between them. The outputs of both calls are produced as outputs from
// the trace and both are written to a .npy file for processing. Inputs can
// also come from an .npy file. See iree-run-module usage for more information
// on the `--input=` and `--output=` flags.

// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | \
// RUN:  iree-run-trace %S/iree-run-trace.yml \
// RUN:                 --device=local-sync \
// RUN:                 --input=4xf32=4,4,4,4 \
// RUN:                 --output=@%t \
// RUN:                 --output=+%t) && \
// RUN:  python3 %S/echo_npy.py %t | \
// RUN: FileCheck %s

//      CHECK{LITERAL}: [ 0. 4. 8. 12.]
// CHECK-NEXT{LITERAL}: [ 0. 12. 24. 36.]

func.func @mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.mulf %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
