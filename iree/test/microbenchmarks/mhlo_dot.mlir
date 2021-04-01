// The following ops are sampled from mobile_bert
// https://github.com/google/iree/blob/main/integrations/tensorflow/e2e/mobile_bert_squad_test.py

func @dot_384x384x512() -> tensor<384x512xf32> attributes { iree.module.export } {
    %lhs = iree.unfoldable_constant dense<1.0> : tensor<384x384xf32>
    %rhs = iree.unfoldable_constant dense<1.0> : tensor<384x512xf32>
    %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<384x384xf32>, tensor<384x512xf32>) -> tensor<384x512xf32>
    return %0: tensor<384x512xf32>
}

func @dot_384x128x128() -> tensor<384x128xf32> attributes { iree.module.export } {
    %lhs = iree.unfoldable_constant dense<1.0> : tensor<384x128xf32>
    %rhs = iree.unfoldable_constant dense<1.0> : tensor<128x128xf32>
    %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<384x128xf32>, tensor<128x128xf32>) -> tensor<384x128xf32>
    return %0 : tensor<384x128xf32>
}

func @dot_384x128x512() -> tensor<384x512xf32> attributes { iree.module.export } {
    %lhs = iree.unfoldable_constant dense<1.0> : tensor<384x128xf32>
    %rhs = iree.unfoldable_constant dense<1.0> : tensor<128x512xf32>
    %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<384x128xf32>, tensor<128x512xf32>) -> tensor<384x512xf32>
    return %0 : tensor<384x512xf32>
}

func @dot_384x512x128() -> tensor<384x128xf32> attributes { iree.module.export } {
    %lhs = iree.unfoldable_constant dense<1.0> : tensor<384x512xf32>
    %rhs = iree.unfoldable_constant dense<1.0> : tensor<512x128xf32>
    %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<384x512xf32>, tensor<512x128xf32>) -> tensor<384x128xf32>
    return %0 : tensor<384x128xf32>
}

func @dot_384x512x2() -> tensor<384x2xf32> attributes { iree.module.export } {
    %lhs = iree.unfoldable_constant dense<1.0> : tensor<384x512xf32> 
    %rhs = iree.unfoldable_constant dense<1.0> : tensor<512x2xf32>
    %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<384x512xf32>, tensor<512x2xf32>) -> tensor<384x2xf32>
    return %0 : tensor<384x2xf32>
}

func @dot_384x384x32() -> tensor<384x32xf32> attributes { iree.module.export } {
    %lhs = iree.unfoldable_constant dense<1.0> : tensor<384x384xf32>
    %rhs = iree.unfoldable_constant dense<1.0> : tensor<384x32xf32>
    %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<384x384xf32>, tensor<384x32xf32>) -> tensor<384x32xf32>
    return %0 : tensor<384x32xf32>
}
