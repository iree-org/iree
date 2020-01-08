// RUN: iree-opt -pass-pipeline='iree-linalg-to-spirv{workgroup-size=2,2 num-workgroups=2,2}' %s

#map0 = (d0, d1) -> (d0, d1)

module {
  func @fmul(%arg0: memref<12x4xf32>, %arg1: memref<12x4xf32>, %arg2: memref<12x4xf32>) {
    linalg.generic {args_in = 2 : i64, args_out = 1 : i64, indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} %arg0, %arg1, %arg2 {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):   // no predecessors
      %0 = mulf %arg3, %arg4 : f32
      linalg.yield %0 : f32
    }: memref<12x4xf32>, memref<12x4xf32>, memref<12x4xf32>
    return
  }
}
