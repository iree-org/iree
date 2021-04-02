
func @pad_test() attributes { iree.module.export } {
    %0 = "tosa.const"() { value = dense<1> : tensor<1x1x1xi32> } : ()  -> (tensor<1x1x1xi32>)
    %1 = "tosa.const"() { value = dense<[[0, 1], [1, 0], [0, 0]]> : tensor<3x2xi32> } : ()  -> (tensor<3x2xi32>)
    %result = "tosa.pad"(%0, %1)  : (tensor<1x1x1xi32>, tensor<3x2xi32>)  -> (tensor<2x2x1xi32>)
    check.expect_eq_const(%result, dense<[[[0], [1]], [[0], [0]]]> : tensor<2x2x1xi32>) : tensor<2x2x1xi32>
    return
}
