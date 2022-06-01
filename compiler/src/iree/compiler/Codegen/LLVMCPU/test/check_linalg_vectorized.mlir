// RUN: iree-opt --iree-llvmcpu-check-linalg-vectorized %s --verify-diagnostics -split-input-file

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  // expected-warning @+1 {{one or more operations were found not vectorized}}
  func.func @dynamic_abs(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map],
                         iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %4 = math.abs %arg1 : f32
      linalg.yield %4 : f32
    } -> tensor<?x?xf32>
    return %3 : tensor<?x?xf32>
  }
}
