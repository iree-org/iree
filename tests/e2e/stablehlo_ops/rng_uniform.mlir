// Note that they are stateless random generators, so they have fixed results.
func.func @rng_uniform_1d() {
    %min = util.unfoldable_constant dense<-10.0> : tensor<f32>
    %max = util.unfoldable_constant dense<10.0> : tensor<f32>
    %shape = util.unfoldable_constant dense<[10]>  : tensor<1xi32>
    %res = "stablehlo.rng"(%min, %max, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<1xi32>) -> tensor<10xf32>
    check.expect_almost_eq_const(%res, dense<[
        -9.99994, -4.8613, 0.277344, 5.41599, -9.44537, -4.30673, 0.831918, 5.97056, -8.8908, -3.75215
        ]> : tensor<10xf32>) : tensor<10xf32>
    return
}

func.func @rng_uniform_2d() {
    %min = util.unfoldable_constant dense<-10.0> : tensor<f32>
    %max = util.unfoldable_constant dense<10.0> : tensor<f32>
    %shape = util.unfoldable_constant dense<[3, 3]>  : tensor<2xi32>
    %res = "stablehlo.rng"(%min, %max, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi32>) -> tensor<3x3xf32>
    check.expect_almost_eq_const(%res, dense<[
        [6.55154, -8.30982, -3.17117],
        [1.75741, 6.89606, -7.9653],
        [-3.03671, 2.10193, 7.24057]]> : tensor<3x3xf32>) : tensor<3x3xf32>
    return
}

func.func @rng_uniform_3d() {
    %min = util.unfoldable_constant dense<-10.0> : tensor<f32>
    %max = util.unfoldable_constant dense<10.0> : tensor<f32>
    %shape = util.unfoldable_constant dense<[2, 2, 2]>  : tensor<3xi32>
    %res = "stablehlo.rng"(%min, %max, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi32>) -> tensor<2x2x2xf32>
    check.expect_almost_eq_const(%res, dense<[
        [[3.04814, 8.18679], [-1.74598, 3.39266]],
        [[-6.91349, -1.77484], [8.29239, -6.56897]]]> : tensor<2x2x2xf32>) : tensor<2x2x2xf32>
    return
}
