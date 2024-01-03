// RUN: iree-opt --iree-check-known-dialects --allow-unregistered-dialect --verify-diagnostics %s

// expected-error@+1 {{one or more unknown operations were found in the compiler input (did you mean to pre-process through an IREE importer frontend?)}}
module {
func.func @unknown_op(%arg0 : tensor<?x8x8x256xf32>) -> tensor<?x16x16x256xf32> {
    %0 = arith.constant dense<16> : tensor<2xi32>
    // expected-note@+1 {{tf.ResizeNearestNeighbor}}
    %1 = "tf.ResizeNearestNeighbor"(%arg0, %0) {align_corners = false, device = "", half_pixel_centers = true} : (tensor<?x8x8x256xf32>, tensor<2xi32>) -> tensor<?x16x16x256xf32>
    return %1 : tensor<?x16x16x256xf32>
}
}
