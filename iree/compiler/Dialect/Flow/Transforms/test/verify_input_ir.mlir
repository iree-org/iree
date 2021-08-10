// RUN: iree-opt -iree-verify-input-legality -verify-diagnostics %s -split-input-file

func @check_no_mlir(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{illegal operation in input to iree core compiler. Use -iree-input-type=mhlo to legalize this operation}}
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func @check_no_tosa(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{illegal operation in input to iree core compiler. Use -iree-input-type=tosa to legalize this operation}}
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
