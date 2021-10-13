// RUN: iree-opt -iree-verify-input-legality -verify-diagnostics %s -split-input-file

// expected-error@below {{illegal operations still remain}}
func @check_no_mhlo(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{illegal op still exists}}
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// expected-error@below {{illegal operations still remain}}
func @check_no_tosa(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{illegal op still exists}}
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
