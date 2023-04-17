// RUN: iree-opt %s --iree-stablehlo-to-linalg --split-input-file \
// RUN:   --canonicalize | FileCheck %s

// RUN: iree-opt %s --iree-stablehlo-to-linalg="enable-primitive-ops=true" \
// RUN:   --split-input-file --canonicalize | \
// RUN:   FileCheck %s --check-prefix=CHECK-PRIMITIVE

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK:     @reduce_add
// CHECK-PRIMITIVE-LABEL: @reduce_add
func.func @reduce_add(%arg0: tensor<5x4xi32>, %arg1: tensor<i32>) -> tensor<5xi32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, someattr} : (tensor<5x4xi32>, tensor<i32>) -> tensor<5xi32>
  func.return %0 : tensor<5xi32>
}
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<5xi32>)
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addi %[[RHS_IN]], %[[LHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-PRIMITIVE-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-PRIMITIVE-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-PRIMITIVE-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK-PRIMITIVE: linalg.reduce { arith.addi }
// CHECK-PRIMITIVE-SAME: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-PRIMITIVE-SAME: outs(%[[FILL_TENSOR]] : tensor<5xi32>)
// CHECK-PRIMITIVE-SAME: dimensions = [1]  {someattr}

// -----

// CHECK-LABEL: @reduce_add_unranked
// CHECK-PRIMITIVE-LABEL: @reduce_add_unranked
func.func @reduce_add_unranked(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, someattr} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}
// CHECK: stablehlo.reduce
// CHECK-PRIMITIVE: stablehlo.reduce

// -----

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK:     @reduce_dim0
// CHECK-PRIMITIVE-LABEL: @reduce_dim0
func.func @reduce_dim0(%arg0: tensor<5x4xi32>, %arg1: tensor<i32>) -> tensor<4xi32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = stablehlo.maximum %arg3, %arg4 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5x4xi32>, tensor<i32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<4xi32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxsi %[[RHS_IN]], %[[LHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-PRIMITIVE-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-PRIMITIVE-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-PRIMITIVE-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK-PRIMITIVE: linalg.reduce { arith.maxsi }
// CHECK-PRIMITIVE-SAME: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-PRIMITIVE-SAME: outs(%[[FILL_TENSOR]] : tensor<4xi32>)
// CHECK-PRIMITIVE-SAME: dimensions = [0]

// -----

func.func @reduce_dynamic_output(%arg0: tensor<5x4xi32>, %arg1: tensor<i32>) -> tensor<?xi32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = stablehlo.maximum %arg3, %arg4 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5x4xi32>, tensor<i32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// Regression test: just check that this lowers successfully.
// CHECK-LABEL: @reduce_dynamic_output
// CHECK: linalg.generic

// CHECK-PRIMITIVE-LABEL: @reduce_dynamic_output
// CHECK-PRIMITIVE: linalg.reduce

// -----

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK:     @reduce_init_const
func.func @reduce_init_const(%arg0: tensor<1x10xf32>) -> tensor<1xf32> {
  %cst = arith.constant dense<0xFF800000> : tensor<f32>
  %0 = "stablehlo.reduce"(%arg0, %cst) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}
// CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<1x10xf32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<1xf32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addf %[[RHS_IN]], %[[LHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK:     @reduce_multi_dimensions
func.func @reduce_multi_dimensions(%arg0: tensor<5x4x3xi32>,
                              %arg1: tensor<i32>) -> tensor<4xi32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<5x4x3xi32>, tensor<i32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4x3xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<4xi32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addi %[[RHS_IN]], %[[LHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: @reduce_lexicographic_min_complex
// CHECK-PRIMITIVE-LABEL: @reduce_lexicographic_min_complex
func.func @reduce_lexicographic_min_complex(%arg0: tensor<?x3x4xcomplex<f64>>,
                                            %arg1: tensor<complex<f64>>)
  -> tensor<complex<f64>> {
  %0 = stablehlo.reduce(%arg0 init: %arg1)
   across dimensions = [0, 1, 2]
   : (tensor<?x3x4xcomplex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
   reducer(%arg3: tensor<complex<f64>>, %arg4: tensor<complex<f64>>)  {
    %1 = stablehlo.real %arg3 : (tensor<complex<f64>>) -> tensor<f64>
    %2 = stablehlo.convert %arg4 : (tensor<complex<f64>>) -> tensor<f64>
    %3 = "stablehlo.compare"(%1, %2)
      {comparison_direction = #stablehlo<comparison_direction EQ>}
      : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %4 = stablehlo.imag %arg3 : (tensor<complex<f64>>) -> tensor<f64>
    %5 = stablehlo.imag %arg4 : (tensor<complex<f64>>) -> tensor<f64>
    %6 = "stablehlo.compare"(%4, %5)
      {comparison_direction = #stablehlo<comparison_direction LT>}
      : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %7 = "stablehlo.compare"(%1, %2)
      {comparison_direction = #stablehlo<comparison_direction LT>}
      : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %8 = "stablehlo.select"(%3, %6, %7)
      : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %9 = "stablehlo.select"(%8, %arg3, %arg4)
      : (tensor<i1>, tensor<complex<f64>>, tensor<complex<f64>>)
      -> tensor<complex<f64>>
    "stablehlo.return"(%9) : (tensor<complex<f64>>) -> ()
  }
  return %0 : tensor<complex<f64>>
}

// CHECK: linalg.generic
// CHECK: complex.re
// CHECK: complex.re
// CHECK: arith.cmpf
// CHECK: complex.im
// CHECK: complex.im
// CHECK: arith.cmpf
// CHECK: arith.cmpf
// CHECK: arith.select

// CHECK-PRIMITIVE: linalg.reduce
// CHECK-PRIMITIVE: complex.re
// CHECK-PRIMITIVE: complex.re
// CHECK-PRIMITIVE: arith.cmpf
// CHECK-PRIMITIVE: complex.im
// CHECK-PRIMITIVE: complex.im
// CHECK-PRIMITIVE: arith.cmpf
// CHECK-PRIMITIVE: arith.cmpf
// CHECK-PRIMITIVE: arith.select

// -----

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK:     func @reduce_dynamic(%[[ARG0:.*]]: tensor<?x?xi32>
func.func @reduce_dynamic(%arg0: tensor<?x?xi32>, %arg1: tensor<i32>) -> tensor<?xi32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = stablehlo.add %arg3, %arg4 : tensor<i32>
    "stablehlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xi32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty(%[[DIM1]])
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<?x?xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<?xi32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addi %[[RHS_IN]], %[[LHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-DAG:  #[[MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG:  #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK:      func @variadic_reduce
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-PRIMITIVE-LABEL: func @variadic_reduce
// CHECK-PRIMITIVE-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-PRIMITIVE-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @variadic_reduce(%arg0: tensor<9x2xi32>, %arg1: tensor<9x2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
  %cst0 = stablehlo.constant dense<-2147483648> : tensor<i32>
  %cst1 = stablehlo.constant dense<0> : tensor<i32>
  %res0, %res1 = "stablehlo.reduce"(%arg0, %arg1, %cst0, %cst1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg15: tensor<i32>, %arg16: tensor<i32>):
    %669 = "stablehlo.compare"(%arg2, %arg15) {comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %670 = "stablehlo.select"(%669, %arg2, %arg15) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %671 = "stablehlo.compare"(%arg2, %arg15) {comparison_direction = #stablehlo<comparison_direction EQ>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %672 = stablehlo.minimum %arg3, %arg16 : tensor<i32>
    %673 = "stablehlo.select"(%669, %arg3, %arg16) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %674 = "stablehlo.select"(%671, %672, %673) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%670, %674) : (tensor<i32>, tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<9x2xi32>, tensor<9x2xi32>, tensor<i32>, tensor<i32>) -> (tensor<2xi32>, tensor<2xi32>)
  func.return %res0, %res1 : tensor<2xi32>, tensor<2xi32>
}
// CHECK-DAG:    %[[CST0:.*]] = arith.constant -2147483648 : i32
// CHECK-DAG:    %[[CST1:.*]] = arith.constant 0 : i32
// CHECK:        %[[INIT0:.*]] = tensor.empty() : tensor<2xi32>
// CHECK:        %[[FILL0:.*]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT0]]
// CHECK:        %[[INIT1:.*]] = tensor.empty() : tensor<2xi32>
// CHECK:        %[[FILL1:.*]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT1]]
// CHECK:        %[[RES:.+]]:2 = linalg.generic {
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<9x2xi32>, tensor<9x2xi32>)
// CHECK-SAME:    outs(%[[FILL0]], %[[FILL1]] : tensor<2xi32>, tensor<2xi32>)
// CHECK-NEXT:   ^bb0(%[[IN0:.*]]: i32, %[[IN1:.*]]: i32, %[[OUT0:.*]]: i32, %[[OUT1:.*]]: i32):
// CHECK-NEXT:     %[[T1:.*]] = arith.cmpi sge, %[[OUT0]], %[[IN0]] : i32
// CHECK-NEXT:     %[[T2:.*]] = arith.select %[[T1]], %[[OUT0]], %[[IN0]] : i32
// CHECK-NEXT:     %[[T3:.*]] = arith.cmpi eq, %[[OUT0]], %[[IN0]] : i32
// CHECK-NEXT:     %[[T4:.*]] = arith.minsi %[[OUT1:.*]], %[[IN1]] : i32
// CHECK-NEXT:     %[[T5:.*]] = arith.select %[[T1]], %[[OUT1]], %[[IN1]] : i32
// CHECK-NEXT:     %[[T6:.*]] = arith.select %[[T3]], %[[T4]], %[[T5]] : i32
// CHECK-NEXT:     linalg.yield %[[T2]], %[[T6]]

// CHECK-PRIMITIVE-DAG:    %[[CST0:.*]] = arith.constant -2147483648 : i32
// CHECK-PRIMITIVE-DAG:    %[[CST1:.*]] = arith.constant 0 : i32
// CHECK-PRIMITIVE:        %[[INIT0:.*]] = tensor.empty() : tensor<2xi32>
// CHECK-PRIMITIVE:        %[[FILL0:.*]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT0]]
// CHECK-PRIMITIVE:        %[[INIT1:.*]] = tensor.empty() : tensor<2xi32>
// CHECK-PRIMITIVE:        %[[FILL1:.*]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT1]]
// CHECK-PRIMITIVE:        %[[RES:.+]]:2 = linalg.reduce
// CHECK-PRIMITIVE-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<9x2xi32>, tensor<9x2xi32>)
// CHECK-PRIMITIVE-SAME:    outs(%[[FILL0]], %[[FILL1]] : tensor<2xi32>, tensor<2xi32>)
// CHECK-PRIMITIVE-SAME:    dimensions = [0]
// CHECK-PRIMITIVE-NEXT:   (%[[IN0:.*]]: i32, %[[IN1:.*]]: i32, %[[OUT0:.*]]: i32, %[[OUT1:.*]]: i32) {
// CHECK-PRIMITIVE-NEXT:     %[[T1:.*]] = arith.cmpi sge, %[[OUT0]], %[[IN0]] : i32
// CHECK-PRIMITIVE-NEXT:     %[[T2:.*]] = arith.select %[[T1]], %[[OUT0]], %[[IN0]] : i32
// CHECK-PRIMITIVE-NEXT:     %[[T3:.*]] = arith.cmpi eq, %[[OUT0]], %[[IN0]] : i32
// CHECK-PRIMITIVE-NEXT:     %[[T4:.*]] = arith.minsi %[[OUT1:.*]], %[[IN1]] : i32
// CHECK-PRIMITIVE-NEXT:     %[[T5:.*]] = arith.select %[[T1]], %[[OUT1]], %[[IN1]] : i32
// CHECK-PRIMITIVE-NEXT:     %[[T6:.*]] = arith.select %[[T3]], %[[T4]], %[[T5]] : i32
// CHECK-PRIMITIVE-NEXT:     linalg.yield %[[T2]], %[[T6]]

// -----

// CHECK-DAG:  #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK:      func @variadic_diff_type_reduce
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-PRIMITIVE-LABEL: func @variadic_diff_type_reduce
// CHECK-PRIMITIVE-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-PRIMITIVE-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @variadic_diff_type_reduce(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xi32>) -> (tensor<128xf32>, tensor<128xi32>) {
  %cst0 = stablehlo.constant dense<1.0> : tensor<f32>
  %cst1 = stablehlo.constant dense<1> : tensor<i32>
  %res0, %res1 = "stablehlo.reduce"(%arg0, %arg1, %cst0, %cst1) ({
  ^bb0(%arg7: tensor<f32>, %arg8: tensor<i32>, %arg9: tensor<f32>, %arg10: tensor<i32>):
    %0 = "stablehlo.compare"(%arg7, %arg9) {comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %1 = "stablehlo.select"(%0, %arg7, %arg9) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "stablehlo.select"(%0, %arg8, %arg10) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "stablehlo.return"(%1, %2) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<128x10xf32>, tensor<128x10xi32>, tensor<f32>, tensor<i32>) ->(tensor<128xf32>, tensor<128xi32>)
  func.return %res0, %res1 : tensor<128xf32>, tensor<128xi32>
}
// CHECK-DAG:        %[[CST0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:        %[[CST1:.*]] = arith.constant 1 : i32
// CHECK:        %[[INIT0:.*]] = tensor.empty() : tensor<128xf32>
// CHECK:        %[[FILL0:.*]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT0]]
// CHECK:        %[[INIT1:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:        %[[FILL1:.*]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT1]]
// CHECK:        %[[RES:.+]]:2 = linalg.generic {
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<128x10xf32>, tensor<128x10xi32>)
// CHECK-SAME:    outs(%[[FILL0]], %[[FILL1]] : tensor<128xf32>, tensor<128xi32>)
// CHECK-NEXT:   ^bb0(%[[LHS0:.*]]: f32, %[[LHS1:.*]]: i32, %[[RHS0:.*]]: f32, %[[RHS1:.*]]: i32):
// CHECK-NEXT:      %[[B0:.*]] = arith.cmpf oge, %[[RHS0]], %[[LHS0]] : f32
// CHECK-NEXT:      %[[RES0:.*]] = arith.select %[[B0]], %[[RHS0]], %[[LHS0]] : f32
// CHECK-NEXT:      %[[RES1:.*]] = arith.select %[[B0]], %[[RHS1]], %[[LHS1]] : i32
// CHECK-NEXT:      linalg.yield %[[RES0]], %[[RES1]] : f32, i32

// CHECK-PRIMITIVE-DAG:        %[[CST0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-PRIMITIVE-DAG:        %[[CST1:.*]] = arith.constant 1 : i32
// CHECK-PRIMITIVE:        %[[INIT0:.*]] = tensor.empty() : tensor<128xf32>
// CHECK-PRIMITIVE:        %[[FILL0:.*]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT0]]
// CHECK-PRIMITIVE:        %[[INIT1:.*]] = tensor.empty() : tensor<128xi32>
// CHECK-PRIMITIVE:        %[[FILL1:.*]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT1]]
// CHECK-PRIMITIVE:        %[[RES:.+]]:2 = linalg.reduce
// CHECK-PRIMITIVE-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<128x10xf32>, tensor<128x10xi32>)
// CHECK-PRIMITIVE-SAME:     outs(%[[FILL0]], %[[FILL1]] : tensor<128xf32>, tensor<128xi32>)
// CHECK-PRIMITIVE-SAME:     dimensions = [1]
// CHECK-PRIMITIVE-NEXT:   (%[[LHS0:.*]]: f32, %[[LHS1:.*]]: i32, %[[RHS0:.*]]: f32, %[[RHS1:.*]]: i32) {
// CHECK-PRIMITIVE-NEXT:      %[[B0:.*]] = arith.cmpf oge, %[[RHS0]], %[[LHS0]] : f32
// CHECK-PRIMITIVE-NEXT:      %[[RES0:.*]] = arith.select %[[B0]], %[[RHS0]], %[[LHS0]] : f32
// CHECK-PRIMITIVE-NEXT:      %[[RES1:.*]] = arith.select %[[B0]], %[[RHS1]], %[[LHS1]] : i32
// CHECK-PRIMITIVE-NEXT:      linalg.yield %[[RES0]], %[[RES1]] : f32, i32
