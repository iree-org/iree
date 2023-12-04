  func.func @forward(%arg0: tensor<?x4096xf16>, %arg1: tensor<4096x32000xf16>) -> tensor<1x1xi64> attributes {hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {ukernels="all"}>} {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c-1_i64 = arith.constant -1 : i64
    %dim = tensor.dim %arg0, %c0 : tensor<?x4096xf16>
    %0 = tensor.empty(%dim) : tensor<?x32000xf16>
    %1 = linalg.fill ins(%cst_0 : f16) outs(%0 : tensor<?x32000xf16>) -> tensor<?x32000xf16>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<?x4096xf16>, tensor<4096x32000xf16>) outs(%1 : tensor<?x32000xf16>) -> tensor<?x32000xf16>
    %expanded = tensor.expand_shape %2 [[0, 1], [2]] : tensor<?x32000xf16> into tensor<1x?x32000xf16>
    %3 = tensor.empty(%dim) : tensor<1x?x32000xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x?x32000xf16>) outs(%3 : tensor<1x?x32000xf32>) {
    ^bb0(%in: f16, %out: f32):
      %20 = arith.extf %in : f16 to f32
      linalg.yield %20 : f32
    } -> tensor<1x?x32000xf32>
    %5 = arith.index_cast %dim : index to i64
    %6 = arith.addi %5, %c-1_i64 : i64
    %7 = arith.addi %6, %5 : i64
    %8 = arith.cmpi sge, %6, %c0_i64 : i64
    %9 = arith.select %8, %6, %7 : i64
    %10 = arith.cmpi slt, %9, %c0_i64 : i64
    %11 = arith.select %10, %c0_i64, %9 : i64
    %12 = arith.cmpi sgt, %11, %5 : i64
    %13 = arith.select %12, %5, %11 : i64
    %14 = arith.index_cast %13 : i64 to index
    %extracted_slice = tensor.extract_slice %4[0, %14, 0] [1, 1, 32000] [1, 1, 1] : tensor<1x?x32000xf32> to tensor<1x1x32000xf32>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x1x32000xf32> into tensor<1x32000xf32>
    %15 = tensor.empty() : tensor<1xi64>
    %16 = linalg.fill ins(%c0_i64 : i64) outs(%15 : tensor<1xi64>) -> tensor<1xi64>
    %17 = tensor.empty() : tensor<1xf32>
    %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<1xf32>) -> tensor<1xf32>
    %19:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%collapsed : tensor<1x32000xf32>) outs(%18, %16 : tensor<1xf32>, tensor<1xi64>) {
    ^bb0(%in: f32, %out: f32, %out_2: i64):
      %20 = linalg.index 1 : index
      %21 = arith.index_cast %20 : index to i64
      %22 = arith.maximumf %in, %out : f32
      %23 = arith.cmpf ogt, %in, %out : f32
      %24 = arith.select %23, %21, %out_2 : i64
      linalg.yield %22, %24 : f32, i64
    } -> (tensor<1xf32>, tensor<1xi64>)
    %expanded_1 = tensor.expand_shape %19#1 [[0, 1]] : tensor<1xi64> into tensor<1x1xi64>
    return %expanded_1 : tensor<1x1xi64>
  }