func.func @pad_1D_test() {
    %0 = util.unfoldable_constant dense<42> : tensor<2xi32>
    %1 = tosa.const_shape { value = dense<[3, 2]> : tensor<2xindex> } : ()  -> !tosa.shape<2>
    %result = tosa.pad %0, %1 : (tensor<2xi32>, !tosa.shape<2>) -> (tensor<7xi32>)
    check.expect_eq_const(%result, dense<[0, 0, 0, 42, 42, 0, 0]> : tensor<7xi32>) : tensor<7xi32>
    return
}

func.func @pad_2D_test() {
    %0 = util.unfoldable_constant dense<42> : tensor<2x2xi32>
    %1 = tosa.const_shape { value = dense<[1, 1, 1, 1]> : tensor<4xindex> } : ()  -> !tosa.shape<4>
    %result = tosa.pad %0, %1 : (tensor<2x2xi32>, !tosa.shape<4>) -> (tensor<4x4xi32>)
    check.expect_eq_const(%result, dense<[[0, 0, 0, 0], [0, 42, 42, 0], [0, 42, 42, 0], [0, 0, 0, 0]]> : tensor<4x4xi32>) : tensor<4x4xi32>
    return
}

func.func @pad_3D_test() {
    %0 = util.unfoldable_constant dense<42> : tensor<1x1x2xi32>
    %1 = tosa.const_shape { value = dense<[0, 1, 1, 0, 0, 0]> : tensor<6xindex> } : ()  -> !tosa.shape<6>
    %result = tosa.pad %0, %1 : (tensor<1x1x2xi32>, !tosa.shape<6>) -> (tensor<2x2x2xi32>)
    check.expect_eq_const(%result, dense<[[[0, 0], [42, 42]], [[0, 0], [0, 0]]]> : tensor<2x2x2xi32>) : tensor<2x2x2xi32>
    return
}
