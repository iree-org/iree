// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

func @ieq(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xi32>) : tensor<4xi32>
  %1 = iree.load_input(%arg1 : memref<4xi32>) : tensor<4xi32>
  // CHECK: spv.IEqual
  %2 = cmpi "eq", %0, %1 : tensor<4xi32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @ineq(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xi32>) : tensor<4xi32>
  %1 = iree.load_input(%arg1 : memref<4xi32>) : tensor<4xi32>
  // CHECK: spv.INotEqual
  %2 = cmpi "ne", %0, %1 : tensor<4xi32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @islt(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xi32>) : tensor<4xi32>
  %1 = iree.load_input(%arg1 : memref<4xi32>) : tensor<4xi32>
  // CHECK: spv.SLessThan
  %2 = cmpi "slt", %0, %1 : tensor<4xi32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @isle(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xi32>) : tensor<4xi32>
  %1 = iree.load_input(%arg1 : memref<4xi32>) : tensor<4xi32>
  // CHECK: spv.SLessThanEqual
  %2 = cmpi "sle", %0, %1 : tensor<4xi32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @isgt(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xi32>) : tensor<4xi32>
  %1 = iree.load_input(%arg1 : memref<4xi32>) : tensor<4xi32>
  // CHECK: spv.SGreaterThan
  %2 = cmpi "sgt", %0, %1 : tensor<4xi32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @isge(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xi32>) : tensor<4xi32>
  %1 = iree.load_input(%arg1 : memref<4xi32>) : tensor<4xi32>
  // CHECK: spv.SGreaterThanEqual
  %2 = cmpi "sge", %0, %1 : tensor<4xi32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @oeq(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: spv.FOrdEqual
  %2 = cmpf "oeq", %0, %1 : tensor<4xf32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @oge(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: spv.FOrdGreaterThanEqual
  %2 = cmpf "oge", %0, %1 : tensor<4xf32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @ogt(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: spv.FOrdGreaterThan
  %2 = cmpf "ogt", %0, %1 : tensor<4xf32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @ole(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: spv.FOrdLessThanEqual
  %2 = cmpf "ole", %0, %1 : tensor<4xf32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @olt(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: spv.FOrdLessThan
  %2 = cmpf "olt", %0, %1 : tensor<4xf32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @ueq(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: spv.FUnordEqual
  %2 = cmpf "ueq", %0, %1 : tensor<4xf32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @uge(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: spv.FUnordGreaterThanEqual
  %2 = cmpf "uge", %0, %1 : tensor<4xf32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @ugt(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: spv.FUnordGreaterThan
  %2 = cmpf "ugt", %0, %1 : tensor<4xf32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @ule(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: spv.FUnordLessThanEqual
  %2 = cmpf "ule", %0, %1 : tensor<4xf32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @ult(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: spv.FUnordLessThan
  %2 = cmpf "ult", %0, %1 : tensor<4xf32>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @beq(%arg0: memref<4xi1>, %arg1: memref<4xi1>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xi1>) : tensor<4xi1>
  %1 = iree.load_input(%arg1 : memref<4xi1>) : tensor<4xi1>
  // CHECK: spv.LogicalEqual
  %2 = cmpi "eq", %0, %1 : tensor<4xi1>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}

// -----

func @bneq(%arg0: memref<4xi1>, %arg1: memref<4xi1>, %arg2: memref<4xi1>)
attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xi1>) : tensor<4xi1>
  %1 = iree.load_input(%arg1 : memref<4xi1>) : tensor<4xi1>
  // CHECK: spv.LogicalNotEqual
  %2 = cmpi "ne", %0, %1 : tensor<4xi1>
  iree.store_output(%2 : tensor<4xi1>, %arg2 : memref<4xi1>)
  iree.return
}
