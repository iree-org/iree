// Note that they are stateless random generators, so they have fixed results.
func.func @rng_uniform_1d() {
    %min = util.unfoldable_constant dense<5.0> : tensor<f32>
    %max = util.unfoldable_constant dense<10.0> : tensor<f32>
    %shape = util.unfoldable_constant dense<[10]>  : tensor<1xi32>
    %res = "stablehlo.rng"(%min, %max, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<1xi32>) -> tensor<10xf32>
    check.expect_almost_eq_const(%res, dense<[
        8.29371, 9.07137, 6.2785, 8.15428, 5.85622, 6.70665, 9.6468, 7.03495, 6.20795, 5.30799
        ]> : tensor<10xf32>) : tensor<10xf32>
    return
}

func.func @rng_uniform_2d() {
    %min = util.unfoldable_constant dense<-10.0> : tensor<f32>
    %max = util.unfoldable_constant dense<10.0> : tensor<f32>
    %shape = util.unfoldable_constant dense<[3, 3]>  : tensor<2xi32>
    %res = "stablehlo.rng"(%min, %max, %shape) {rng_distribution = #stablehlo<rng_distribution UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi32>) -> tensor<3x3xf32>
    check.expect_almost_eq_const(%res, dense<[
        [3.17482, -4.88602, -6.57512],
        [6.28547, 2.6171, -3.1734],
        [8.58719, -5.16822, 4.12755]]> : tensor<3x3xf32>) : tensor<3x3xf32>
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
