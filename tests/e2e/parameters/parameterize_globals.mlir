// RUN: iree-opt --pass-pipeline="builtin.module(iree-io-parameterize-globals{archive-path="%p/opt.irpa" parameter-namespace=opt})" %s | FileCheck %s

// CHECK-LABEL: module @parameter_example
module @parameter_example {
// CHECK-DAG: util.global private @"__iree_flow___sm_node17__model.layer-1.kernel" = #stream.parameter.named<"opt"::"__iree_flow___sm_node17__model.layer-1.kernel"> : tensor<2x2xf32>
// CHECK-DAG: util.global private @"__iree_flow___sm_node18__model.layer-1.bias" = #stream.parameter.named<"opt"::"__iree_flow___sm_node18__model.layer-1.bias"> : tensor<1x2xf32>
// CHECK-DAG: util.global private @"__iree_flow___sm_node24__model.layer-2.kernel" = #stream.parameter.named<"opt"::"__iree_flow___sm_node24__model.layer-2.kernel"> : tensor<2x2xf32>
// CHECK-DAG: util.global private @"__iree_flow___sm_node25__model.layer-2.bias" = #stream.parameter.named<"opt"::"__iree_flow___sm_node25__model.layer-2.bias"> : tensor<1x2xf32>
  util.global private @"__iree_flow___sm_node17__model.layer-1.kernel" = dense<"0x0000803F000000400000404000008040"> : tensor<2x2xf32>
  util.global private @"__iree_flow___sm_node18__model.layer-1.bias" = dense<"0x0000A0400000C040"> : tensor<1x2xf32>
  util.global private @"__iree_flow___sm_node24__model.layer-2.kernel" = dense<"0x0000E040000000410000104100002041"> : tensor<2x2xf32>
  util.global private @"__iree_flow___sm_node25__model.layer-2.bias" = dense<[[11.0, 12.0]]> : tensor<1x2xf32>
  func.func @predict(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
    %ptr___iree_flow___sm_node17__model.layer-1.kernel = util.global.address @"__iree_flow___sm_node17__model.layer-1.kernel" : !util.ptr<tensor<2x2xf32>>
    %ptr___iree_flow___sm_node18__model.layer-1.bias = util.global.address @"__iree_flow___sm_node18__model.layer-1.bias" : !util.ptr<tensor<1x2xf32>>
    %ptr___iree_flow___sm_node24__model.layer-2.kernel = util.global.address @"__iree_flow___sm_node24__model.layer-2.kernel" : !util.ptr<tensor<2x2xf32>>
    %ptr___iree_flow___sm_node25__model.layer-2.bias = util.global.address @"__iree_flow___sm_node25__model.layer-2.bias" : !util.ptr<tensor<1x2xf32>>
    %0 = arith.constant dense<0.000000e+00> : tensor<1x2xf32>
    %1 = arith.constant dense<0xFF800000> : tensor<f32>
    %2 = arith.constant dense<0.000000e+00> : tensor<f32>
    %cst = arith.constant 0.000000e+00 : f32
    %3 = util.global.load.indirect %ptr___iree_flow___sm_node25__model.layer-2.bias : !util.ptr<tensor<1x2xf32>> -> tensor<1x2xf32>
    %4 = util.global.load.indirect %ptr___iree_flow___sm_node24__model.layer-2.kernel : !util.ptr<tensor<2x2xf32>> -> tensor<2x2xf32>
    %5 = util.global.load.indirect %ptr___iree_flow___sm_node18__model.layer-1.bias : !util.ptr<tensor<1x2xf32>> -> tensor<1x2xf32>
    %6 = util.global.load.indirect %ptr___iree_flow___sm_node17__model.layer-1.kernel : !util.ptr<tensor<2x2xf32>> -> tensor<2x2xf32>
    %empty = tensor.empty() : tensor<1x2xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
    %8 = linalg.matmul ins(%arg0, %6 : tensor<1x2xf32>, tensor<2x2xf32>) outs(%fill : tensor<1x2xf32>) -> tensor<1x2xf32>
    %10 = linalg.add ins(%8, %5 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
    %12 = linalg.matmul ins(%10, %4 : tensor<1x2xf32>, tensor<2x2xf32>) outs(%fill : tensor<1x2xf32>) -> tensor<1x2xf32>
    %14 = linalg.add ins(%12, %3 : tensor<1x2xf32>, tensor<1x2xf32>) outs(%empty : tensor<1x2xf32>) -> tensor<1x2xf32>
    return %14 : tensor<1x2xf32>
  }
}

// RUN: iree-opt --pass-pipeline="builtin.module(iree-io-parameterize-globals{archive-path="%p/compile.irpa" parameter-namespace=compile})" %s | \
// RUN: iree-compile - \
// RUN:   --iree-hal-target-backends=vmvx | \
// RUN: iree-run-module --device=local-task --module=- \
// RUN:   --input=1x2xf32=1.0 \
// RUN:   --parameters=compile=%p/compile.irpa \
// RUN:   --function=predict | \
// RUN:   FileCheck %s --check-prefix=EXEC

// EXEC-LABEL: EXEC @predict
// EXEC: 1x2xf32=[182 204]
