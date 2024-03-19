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
  }) {dimensions = array<i64: 1>, someattr} : (tensor<5x4xi32>, tensor<i32>) -> tensor<5xi32>
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
// CHECK-PRIMITIVE: linalg.reduce { arith.addi {overflowFlags = #arith.overflow<none>} }
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
  }) {dimensions = array<i64: 1>, someattr} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
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
  }) {dimensions = array<i64: 0>} : (tensor<5x4xi32>, tensor<i32>) -> tensor<4xi32>
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
  }) {dimensions = array<i64: 0>} : (tensor<5x4xi32>, tensor<i32>) -> tensor<?xi32>
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
  }) {dimensions = array<i64: 1>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
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
  }) {dimensions = array<i64: 0, 2>} : (tensor<5x4x3xi32>, tensor<i32>) -> tensor<4xi32>
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
  }) {dimensions = array<i64: 1>} : (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
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
  }) {dimensions = array<i64: 0>} : (tensor<9x2xi32>, tensor<9x2xi32>, tensor<i32>, tensor<i32>) -> (tensor<2xi32>, tensor<2xi32>)
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
  }) {dimensions = array<i64: 1>} : (tensor<128x10xf32>, tensor<128x10xi32>, tensor<f32>, tensor<i32>) ->(tensor<128xf32>, tensor<128xi32>)
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

// -----

// Make sure we do not crash on unsupported reductions.

// CHECK-LABEL: func.func @reduce_noop
// CHECK:         stablehlo.reduce
// CHECK-PRIMITIVE-LABEL: func.func @reduce_noop
// CHECK-PRIMITIVE:         stablehlo.reduce
func.func @reduce_noop(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) across dimensions = [] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
    reducer(%arg1: tensor<f32>, %arg2: tensor<f32>) {
    %4 = stablehlo.add %arg1, %arg2 : tensor<f32>
    stablehlo.return %4 : tensor<f32>
  }
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reduce_zero_ext
// CHECK:         stablehlo.reduce
// CHECK-PRIMITIVE-LABEL: func.func @reduce_zero_ext
// CHECK-PRIMITIVE:         stablehlo.reduce
func.func @reduce_zero_ext(%arg0: tensor<0xi1>) -> tensor<i32> {
  %0 = stablehlo.constant dense<false> : tensor<i1>
  %1 = stablehlo.constant dense<false> : tensor<0xi1>
  %2 = stablehlo.compare  NE, %arg0, %1, UNSIGNED : (tensor<0xi1>, tensor<0xi1>) -> tensor<0xi1>
  %3 = stablehlo.convert %2 : (tensor<0xi1>) -> tensor<0xi32>
  %4 = stablehlo.constant dense<0> : tensor<i32>
  %5 = stablehlo.reduce(%3 init: %4) across dimensions = [0] : (tensor<0xi32>, tensor<i32>) -> tensor<i32>
    reducer(%arg1: tensor<i32>, %arg2: tensor<i32>)  {
    %6 = stablehlo.add %arg1, %arg2 : tensor<i32>
    stablehlo.return %6 : tensor<i32>
  }
  return %5 : tensor<i32>
}

// -----

// CHECK-LABEL: func @reduce_window_min_nhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @reduce_window_min_nhwc(%arg0: tensor<1x17x17x64xf32>,
                             %arg1: tensor<f32>) -> tensor<1x8x8x64xf32>{
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = stablehlo.minimum %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 3, 3, 1>,
      window_strides = array<i64: 1, 2, 2, 1>,
      someattr} : (tensor<1x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  func.return %0 : tensor<1x8x8x64xf32>
}
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_nhwc_min
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       someattr,
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>

// -----

// CHECK-LABEL: func @reduce_window_max_nhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @reduce_window_max_nhwc(%arg0: tensor<1x17x17x64xf32>,
                             %arg1: tensor<f32>) -> tensor<1x8x8x64xf32>{
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 3, 3, 1>,
      window_strides = array<i64: 1, 2, 2, 1>} : (tensor<1x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  func.return %0 : tensor<1x8x8x64xf32>
}
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_nhwc_max
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>

// -----

// CHECK-LABEL: func @reduce_window_sum_nhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @reduce_window_sum_nhwc(%arg0: tensor<1x17x17x64xf32>,
                             %arg1: tensor<f32>) -> tensor<1x8x8x64xf32>{
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 3, 3, 1>,
      window_strides = array<i64: 1, 2, 2, 1>} : (tensor<1x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  func.return %0 : tensor<1x8x8x64xf32>
}
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_nhwc_sum
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>

// -----

// CHECK-LABEL: func @reduce_window_max_nhwc_with_cst
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
func.func @reduce_window_max_nhwc_with_cst(%arg0: tensor<1x17x17x64xf32>) -> tensor<1x8x8x64xf32> {
  %0 = arith.constant dense<0xFF800000> : tensor<f32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2 : tensor<f32>):
    %2 = stablehlo.maximum %arg1, %arg2 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 3, 3, 1>,
      window_strides = array<i64: 1, 2, 2, 1>} : (tensor<1x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  func.return %1 : tensor<1x8x8x64xf32>
}

// CHECK-DAG:     %[[CST:.+]] = arith.constant 0xFF800000
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x64xf32
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_nhwc_max
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>

// -----

// CHECK-LABEL: func @reduce_window_sum_max_nhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @reduce_window_sum_max_nhwc(%arg0: tensor<1x17x17x64xf32>,
                             %arg1: tensor<f32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) {
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg0, %arg1, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>, %arg4: tensor<f32>, %arg5 : tensor<f32>):
    %1 = stablehlo.add %arg2, %arg4 : tensor<f32>
    %2 = stablehlo.maximum %arg3, %arg5 : tensor<f32>
    "stablehlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 3, 3, 1>,
      window_strides = array<i64: 1, 2, 2, 1>} : (tensor<1x17x17x64xf32>, tensor<1x17x17x64xf32>, tensor<f32>, tensor<f32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>)
  func.return %0#0, %0#1 : tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>
}

// CHECK:         %[[WINDOW0:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT0:.+]] = tensor.empty() : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL0:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT0]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES0:.+]] = linalg.pooling_nhwc_sum
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW0]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL0]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[WINDOW1:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT1:.+]] = tensor.empty() : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL1:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL1:.+]] = linalg.fill ins(%[[INIT_VAL1]] : f32) outs(%[[INIT1]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES1:.+]] = linalg.pooling_nhwc_max
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW1]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL1]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         return %[[RES0]], %[[RES1]]

// -----

// Just check that this lowers successfully.
// CHECK-LABEL: func @reduce_window_unsigned
func.func @reduce_window_unsigned(%arg0: tensor<1x1xui32>) -> tensor<1x1xui32> {
  %0 = stablehlo.constant dense<0> : tensor<ui32>
  %1 = "stablehlo.reduce_window"(%arg0, %0) ({
  ^bb0(%arg1: tensor<ui32>, %arg2: tensor<ui32>):
    stablehlo.return %arg1 : tensor<ui32>
  }) {
    window_dimensions = array<i64: 1, 1>,
    window_strides = array<i64: 1, 1>
  } : (tensor<1x1xui32>, tensor<ui32>) -> tensor<1x1xui32>
  return %1 : tensor<1x1xui32>
}

// -----

// CHECK-LABEL: func @dynamic_reduce_window_sum_nhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @dynamic_reduce_window_sum_nhwc(%arg0: tensor<?x?x?x?xf32>,
                                      %arg1: tensor<f32>) -> tensor<?x?x?x?xf32>{
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 3, 3, 1>,
      window_strides = array<i64: 1, 2, 2, 1>} : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T1:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T2:.+]] = arith.subi %[[T1]], %[[C3]]
// CHECK:         %[[T3:.+]] = arith.divui %[[T2]], %[[C2]]
// CHECK:         %[[D1:.+]] = arith.addi %[[T3]], %[[C1]]
// CHECK:         %[[T1:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T2:.+]] = arith.subi %[[T1]], %[[C3]]
// CHECK:         %[[T3:.+]] = arith.divui %[[T2]], %[[C2]]
// CHECK:         %[[D2:.+]] = arith.addi %[[T3]], %[[C1]]
// CHECK:         %[[D3:.+]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]], %[[D3]]) : tensor<?x?x?x?xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_nhwc_sum
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<?x?x?x?xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

// -----

// CHECK-LABEL: func @reduce_window_min_ndhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @reduce_window_min_ndhwc(%arg0: tensor<1x17x17x17x64xf32>,
                              %arg1: tensor<f32>) -> tensor<1x8x8x8x64xf32>{
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = stablehlo.minimum %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 3, 3, 3, 1>,
      window_strides = array<i64: 1, 2, 2, 2, 1>} : (tensor<1x17x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x8x64xf32>
  func.return %0 : tensor<1x8x8x8x64xf32>
}
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_ndhwc_min
// CHECK-SAME:      {dilations = dense<1> : vector<3xi64>
// CHECK-SAME:       strides = dense<2> : vector<3xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x17x64xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>

// -----

// CHECK-LABEL: func @reduce_window_max_ndhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @reduce_window_max_ndhwc(%arg0: tensor<1x17x17x17x64xf32>,
                              %arg1: tensor<f32>) -> tensor<1x8x8x8x64xf32>{
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 3, 3, 3, 1>,
      window_strides = array<i64: 1, 2, 2, 2, 1>} : (tensor<1x17x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x8x64xf32>
  func.return %0 : tensor<1x8x8x8x64xf32>
}
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_ndhwc_max
// CHECK-SAME:      {dilations = dense<1> : vector<3xi64>
// CHECK-SAME:       strides = dense<2> : vector<3xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x17x64xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>

// -----

// CHECK-LABEL: func @reduce_window_sum_ndhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @reduce_window_sum_ndhwc(%arg0: tensor<1x17x17x17x64xf32>,
                              %arg1: tensor<f32>) -> tensor<1x8x8x8x64xf32>{
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 3, 3, 3, 1>,
      window_strides = array<i64: 1, 2, 2, 2, 1>} : (tensor<1x17x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x8x64xf32>
  func.return %0 : tensor<1x8x8x8x64xf32>
}
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_ndhwc_sum
// CHECK-SAME:      {dilations = dense<1> : vector<3xi64>
// CHECK-SAME:       strides = dense<2> : vector<3xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x17x64xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>

// -----

// CHECK-LABEL: func @reduce_window_sum_ndhwc_dilated_base
// CHECK: linalg.generic
func.func @reduce_window_sum_ndhwc_dilated_base(
    %arg0: tensor<1x17x17x17x64xf32>,
    %arg1: tensor<f32>) -> tensor<1x8x8x16x64xf32>{
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {base_dilations = array<i64: 1, 1, 1, 2, 1>,
      window_dimensions = array<i64: 1, 3, 3, 3, 1>,
      window_strides = array<i64: 1, 2, 2, 2, 1>} : (tensor<1x17x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x16x64xf32>
  func.return %0 : tensor<1x8x8x16x64xf32>
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2, d1 + d2 * 2)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK:      func @reduce_window_generic
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
func.func @reduce_window_generic(%arg0: tensor<4x6xf32>, %arg1: tensor<f32>) -> tensor<4x7xf32> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {base_dilations = array<i64: 1, 1>, padding = dense<[[0, 3], [1, 2]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 2>, window_dimensions = array<i64: 1, 2>, window_strides = array<i64: 2, 1>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<4x7xf32>
  func.return %0 : tensor<4x7xf32>
}
// CHECK: %[[INIT:.+]] = tensor.empty() : tensor<4x7xf32>
// CHECK: %[[FILL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%[[INIT]] : tensor<4x7xf32>)
// CHECK: ^{{[a-z0-9_]*}}
// CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]*]]: f32
// CHECK:   linalg.yield %[[IN]] : f32

// CHECK: %[[PADVAL:.+]] = tensor.extract %arg1[] : tensor<f32>
// CHECK: %[[PAD:.+]] = tensor.pad %arg0 low[0, 1] high[3, 2]
// CHECK: ^{{[a-z0-9_]*}}
// CHECK-SAME: %{{[a-zA-Z0-9_]*}}: index
// CHECK-SAME: %{{[a-zA-Z0-9_]*}}: index
// CHECK:   tensor.yield %[[PADVAL]] : f32

// CHECK: %[[WINDOW:.+]] = tensor.empty() : tensor<2xf32>
// CHECK: %[[REDUCE:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[PAD]], %[[WINDOW]] : tensor<7x9xf32>, tensor<2xf32>) outs(%[[FILL]] : tensor<4x7xf32>) {
// CHECK: ^{{[a-z0-9_]*}}
// CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
// CHECK-SAME: %[[IN2:[a-zA-Z0-9_]*]]: f32
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]*]]: f32
// CHECK:   %[[ADD:.+]] = arith.addf %[[OUT]], %[[IN]] : f32
// CHECK:   linalg.yield %[[ADD]]

// CHECK: return %[[REDUCE]]
// -----

// CHECK-LABEL: func @reduce_window_generic_captured_constant
func.func @reduce_window_generic_captured_constant(%arg0: tensor<4x6xf32>, %arg1: tensor<f32>) -> tensor<4x7xf32> {
  %c2 = stablehlo.constant dense<2.0> : tensor<f32>
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    %2 = stablehlo.multiply %1, %c2 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {base_dilations = array<i64: 1, 1>, padding = dense<[[0, 3], [1, 2]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 2>, window_dimensions = array<i64: 1, 2>, window_strides = array<i64: 2, 1>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<4x7xf32>
  func.return %0 : tensor<4x7xf32>
}

// CHECK: %[[C2:.*]] = arith.constant 2.0
// CHECK: linalg.generic
// CHECK: %[[SUM:.*]] = arith.addf
// CHECK: %[[PROD:.*]] = arith.mulf %[[SUM]], %[[C2]]
// CHECK: linalg.yield %[[PROD]]

// -----

// CHECK-LABEL: func @reduce_window_generic_padding
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @reduce_window_generic_padding(%arg0: tensor<3x6xf32>, %arg1: tensor<f32>) -> tensor<3x7xf32> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {padding = dense<[[0, 3], [1, 2]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 2>, window_dimensions = array<i64: 1, 2>, window_strides = array<i64: 2, 1>} : (tensor<3x6xf32>, tensor<f32>) -> tensor<3x7xf32>
  func.return %0 : tensor<3x7xf32>
}
// CHECK: %[[PADVAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK: %[[PAD:.+]] = tensor.pad %[[ARG0]] low[0, 1] high[3, 2]
// CHECK: tensor.yield %[[PADVAL]] : f32

// -----

// CHECK-LABEL: func @reduce_window_generic_base_dilation
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @reduce_window_generic_base_dilation(%arg0: tensor<3x6xf32>, %arg1: tensor<f32>) -> tensor<3x4xf32> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {base_dilations = array<i64: 2, 1>, window_dilations = array<i64: 1, 2>, window_dimensions = array<i64: 1, 2>, window_strides = array<i64: 2, 1>} : (tensor<3x6xf32>, tensor<f32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}
// CHECK: %[[PADVAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK: %[[INIT:.+]] = tensor.empty() : tensor<5x6xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[PADVAL]] : f32) outs(%[[INIT]] : tensor<5x6xf32>) -> tensor<5x6xf32>
// CHECK: %[[PAD:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][0, 0] [3, 6] [2, 1] : tensor<3x6xf32> into tensor<5x6xf32>

// -----

// CHECK-LABEL: func @reduce_window_generic_padding_base_dilation
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
func.func @reduce_window_generic_padding_base_dilation(%arg0: tensor<3x6xf32>, %arg1: tensor<f32>) -> tensor<4x7xf32> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {base_dilations = array<i64: 2, 1>, padding = dense<[[0, 3], [1, 2]]> : tensor<2x2xi64>, window_dilations = array<i64: 1, 2>, window_dimensions = array<i64: 1, 2>, window_strides = array<i64: 2, 1>} : (tensor<3x6xf32>, tensor<f32>) -> tensor<4x7xf32>
  func.return %0 : tensor<4x7xf32>
}
// CHECK: %[[PADVAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK: %[[INIT:.+]] = tensor.empty() : tensor<8x9xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[PADVAL]] : f32) outs(%[[INIT]] : tensor<8x9xf32>) -> tensor<8x9xf32>
// CHECK: %[[PAD:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][0, 1] [3, 6] [2, 1] : tensor<3x6xf32> into tensor<8x9xf32>

// -----

// CHECK: #[[MAP:.+]] = affine_map<() -> ()>
// CHECK: func @reduce_window_generic_scalar
func.func @reduce_window_generic_scalar(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = stablehlo.add %arg2, %arg3 : tensor<f32>
    "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {base_dilations = array<i64>, padding = dense<> : tensor<0x2xi64>, window_dilations = array<i64>, window_dimensions = array<i64>, window_strides = array<i64>} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK: linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
