// RUN: (iree-run-mlir -iree-hal-target-backends=vmla -input-value="10xi32=[0,1,2,3,4,5,6,7,8,9]" -input-value="10xi32=[0,1,2,3,4,5,6,7,8,9]" -input-value="10xi32=[0,1,2,3,4,5,6,7,8,9]" -input-value="10xi32=[9,8,7,6,5,4,3,2,1,0]" %s) | IreeFileCheck %s

// CHECK: EXEC @main
// CHECK: 10xi32=9 8 7 6 5 4 3 2 1 0

func @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>, %arg2: tensor<?xi32>, %arg3: tensor<?xi32>) -> tensor<?xi32> attributes {iree.module.export} {
    %1 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "LT"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi1>
    %2 = "mhlo.select"(%1, %arg2, %arg3) : (tensor<?xi1>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %2 : tensor<?xi32>
}

