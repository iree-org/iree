func.func public @cholesky_test() -> () {
  %m = util.unfoldable_constant dense<[
       [4.19151945, 0.70104229, 0.69793355, 0.73441076],
       [0.70104229, 4.27259261, 0.57619844, 0.7572871 ],
       [0.69793355, 0.57619844, 4.35781727, 0.43562294],
       [0.73441076, 0.7572871 , 0.43562294, 4.56119619]]> : tensor<4x4xf32>
  %result = stablehlo.cholesky %m, lower = true : tensor<4x4xf32>
  check.expect_almost_eq_const(%result, dense<[
    [2.0473201,  0.,         0.,         0.        ],
    [0.34241948, 2.0384655,  0.,         0.        ],
    [0.34090105, 0.22539862, 2.0471442,  0.        ],
    [0.3587181,  0.31124148, 0.11879093, 2.0788302 ]]> : tensor<4x4xf32>): tensor<4x4xf32>
  return
}

func.func public @householder_upper_test() -> () {
  %m = util.unfoldable_constant dense<[
       [4.19151945, 0.70104229, 0.69793355, 0.73441076],
       [0.70104229, 4.27259261, 0.57619844, 0.7572871 ],
       [0.69793355, 0.57619844, 4.35781727, 0.43562294],
       [0.73441076, 0.7572871 , 0.43562294, 4.56119619]]> : tensor<4x4xf32>
  %result = stablehlo.cholesky %m, lower = false : tensor<4x4xf32>
  check.expect_almost_eq_const(%result, dense<[
    [2.04732, 0.34241948, 0.34090105, 0.35871810],
    [0.,      2.0384655,  0.22539862, 0.31124148],
    [0.,      0.,         2.0471442,  0.11879092],
    [0.,      0.,         0.,         2.07883020]]> : tensor<4x4xf32>): tensor<4x4xf32>
  return
}
