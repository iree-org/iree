// RUN: iree-opt -split-input-file -iree-mhlo-to-linalg-on-tensors -canonicalize %s | IreeFileCheck %s

func @rng_uniform_1d(%min: tensor<f32>, %max: tensor<f32>) -> tensor<10xf32>
{
  %shape = constant dense<[10]>  : tensor<1xi32>
  %0 = "mhlo.rng_uniform"(%min, %max, %shape) : (tensor<f32>, tensor<f32>, tensor<1xi32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>
}
// CHECK-LABEL: func @rng_uniform_1d
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = muli %[[IDX0_CAST]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL2_CAST:.+]] = uitofp %[[VAL2]] : i32 to f32
// CHECK-DAG:    %[[VAL4:.+]] = mulf %[[VAL2_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL5:.+]] = addf %[[VAL4]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL5]] : f32
// CHECK-NEXT: -> tensor<10xf32>

// -----

func @rng_uniform_2d(%min: tensor<f32>, %max: tensor<f32>) -> tensor<3x3xf32>
{
        %shape = constant dense<[3, 3]>  : tensor<2xi32>
        %0 = "mhlo.rng_uniform"(%min, %max, %shape) : (tensor<f32>, tensor<f32>, tensor<2xi32>) -> tensor<3x3xf32>
        return %0 : tensor<3x3xf32>
}
// CHECK-LABEL: func @rng_uniform_2d
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = index_cast %[[IDX0]] : index to i32
// CHECK-DAG:  %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:  %[[IDX1_CAST:.+]] = index_cast %[[IDX1]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = muli %[[IDX0_CAST]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL3:.+]] = addi %[[IDX1_CAST]], %[[VAL2]] : i32
// CHECK-DAG:    %[[VAL4:.+]] = muli %[[VAL3]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL5:.+]] = addi %[[VAL4]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL5_CAST:.+]] = uitofp %[[VAL5]] : i32 to f32
// CHECK-DAG:    %[[VAL6:.+]] = mulf %[[VAL5_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL7:.+]] = addf %[[VAL6]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL7]] : f32
// CHECK-NEXT: -> tensor<3x3xf32>

// -----

func @rng_uniform_3d(%min: tensor<f32>, %max: tensor<f32>) -> tensor<2x2x2xf32>
{
        %shape = constant dense<[2, 2, 2]>  : tensor<3xi32>
        %0 = "mhlo.rng_uniform"(%min, %max, %shape) : (tensor<f32>, tensor<f32>, tensor<3xi32>) -> tensor<2x2x2xf32>
        return %0 : tensor<2x2x2xf32>
}
// CHECK-LABEL: func @rng_uniform_3d
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = index_cast %[[IDX0]] : index to i32
// CHECK-DAG:  %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:  %[[IDX1_CAST:.+]] = index_cast %[[IDX1]] : index to i32
// CHECK-DAG:  %[[IDX2:.+]] = linalg.index 2 : index
// CHECK-DAG:  %[[IDX2_CAST:.+]] = index_cast %[[IDX2]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = muli %[[IDX0_CAST]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL3:.+]] = addi %[[IDX1_CAST]], %[[VAL2]] : i32
// CHECK-DAG:    %[[VAL4:.+]] = muli %[[VAL3]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL5:.+]] = addi %[[VAL4]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL6:.+]] = addi %[[IDX2_CAST]], %[[VAL5]] : i32
// CHECK-DAG:    %[[VAL7:.+]] = muli %[[VAL6]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL8:.+]] = addi %[[VAL7]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL8_CAST:.+]] = uitofp %[[VAL8]] : i32 to f32
// CHECK-DAG:    %[[VAL6:.+]] = mulf %[[VAL8_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL7:.+]] = addf %[[VAL6]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL7]] : f32
// CHECK-NEXT: -> tensor<2x2x2xf32>

// -----

func @rng_uniform_dynamic_1d(%min: tensor<f32>, %max: tensor<f32>, %shape: tensor<1xi32>) -> tensor<?xf32>
{
  %0 = "mhlo.rng_uniform"(%min, %max, %shape) : (tensor<f32>, tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @rng_uniform_dynamic_1d
// CHECK-DAG:    %[[C0:.+]] = constant 0 : index
// CHECK-DAG:    %[[EXT:.+]] = tensor.extract %{{.+}}[%[[C0]]] : tensor<1xi32>
// CHECK-DAG:    %[[IND:.+]] = index_cast %[[EXT]] : i32 to index
// CHECK-DAG:    %{{.+}} = linalg.init_tensor [%[[IND]]] : tensor<?xf32>
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = constant 2.32830644E-10 : f32
// CHECK-DAG:    %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:    %[[IDX0_CAST:.+]] = index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = muli %[[IDX0_CAST]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL2_CAST:.+]] = uitofp %[[VAL2]] : i32 to f32
// CHECK-DAG:    %[[VAL4:.+]] = mulf %[[VAL2_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL5:.+]] = addf %[[VAL4]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL5]] : f32
// CHECK-NEXT: -> tensor<?xf32>
