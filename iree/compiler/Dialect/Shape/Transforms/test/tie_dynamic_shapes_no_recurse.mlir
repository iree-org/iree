// RUN: iree-opt -allow-unregistered-dialect -split-input-file -verify-diagnostics -iree-shape-tie-dynamic='do-not-recurse-op-names=my_dialect.region1,my_dialect.region2' %s | IreeFileCheck %s

// CHECK-LABEL: @doTie
func @doTie(%arg1 : tensor<?xf32>) {
  // CHECK: shapex.tie_shape
  %0 = "my_dialect.munge"(%arg1) : (tensor<?xf32>) -> tensor<?xf32>
  return
}

// CHECK-LABEL: @noTieDoNotRecurse
func @noTieDoNotRecurse(%arg1 : tensor<?xf32>) {
  // CHECK: my_dialect.region1
  "my_dialect.region1"() ( {
    // CHECK: my_dialect.munge
    %0 = "my_dialect.munge"(%arg1) : (tensor<?xf32>) -> tensor<?xf32>
    // CHECK-NOT: shapex.tie_shape
    // CHECK: my_dialect.terminator
    "my_dialect.terminator"() : () -> ()
  }) : () -> ()
  // CHECK: my_dialect.region2
  "my_dialect.region2"() ( {
    // CHECK: my_dialect.munge
    %1 = "my_dialect.munge"(%arg1) : (tensor<?xf32>) -> tensor<?xf32>
    // CHECK-NOT: shapex.tie_shape
    // CHECK: my_dialect.terminator
    "my_dialect.terminator"() : () -> ()
  }) : () -> ()
  // CHECK: my_dialect.region3
  "my_dialect.region3"() ( {
    // CHECK: my_dialect.munge
    %1 = "my_dialect.munge"(%arg1) : (tensor<?xf32>) -> tensor<?xf32>
    // CHECK: shapex.tie_shape
    // CHECK: my_dialect.terminator
    "my_dialect.terminator"() : () -> ()
  }) : () -> ()
  return
}
