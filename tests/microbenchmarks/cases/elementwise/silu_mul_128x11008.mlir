// SwiGLU-style gate: silu(a) * b, two inputs, 128x11008 f32.

func.func @silu_mul_128x11008(%a: tensor<128x11008xf32>, %b: tensor<128x11008xf32>) -> tensor<128x11008xf32> {
  %e = tensor.empty() : tensor<128x11008xf32>
  %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<128x11008xf32>, tensor<128x11008xf32>) outs(%e : tensor<128x11008xf32>) {
  ^bb0(%x: f32, %y: f32, %out: f32):
    %one = arith.constant 1.0 : f32
    %n = arith.negf %x : f32
    %ex = math.exp %n : f32
    %d = arith.addf %ex, %one : f32
    %sig = arith.divf %one, %d : f32
    %s = arith.mulf %x, %sig : f32
    %m = arith.mulf %s, %y : f32
    linalg.yield %m : f32
  } -> tensor<128x11008xf32>
  return %r : tensor<128x11008xf32>
}
