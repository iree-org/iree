// RUN: iree-opt -split-input-file -convert-iree-to-spirv -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | FileCheck %s

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<42 x i32 [4]> [168]> [0]>, StorageBuffer>
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<42 x i32 [4]> [168]> [0]>, StorageBuffer>
  // CHECK: func [[FN:@simple_load_store_entry_dispatch_0]]()
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = "xla_hlo.copy"(%0) : (tensor<12x42xi32>) -> tensor<12x42xi32>
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    // CHECK: spv.Return
    iree.return
  }
  // CHECK: spv.EntryPoint "GLCompute" [[FN]], [[GLOBALIDVAR]]
}

// -----

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  // CHECK: spv.globalVariable [[ARG2VAR:@.*]] bind(0, 2)
  func @simple_mul_entry_dispatch_0(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL1:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1LOADPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL2:%.*]] = spv.Load "StorageBuffer" [[ARG1LOADPTR]]
    %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
    // CHECK: [[RESULT:%.*]] = spv.FMul [[VAL1]], [[VAL2]]
    %2 = mulf %0, %1 : tensor<4xf32>
    // CHECK: [[ARG2PTR:%.*]] = spv._address_of [[ARG2VAR]]
    // CHECK: [[ZERO3:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG2STOREPTR:%.*]] = spv.AccessChain [[ARG2PTR]]{{\[}}[[ZERO3]], [[GLOBALIDX]]{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG2STOREPTR]], [[RESULT]]
    iree.store_output(%2 : tensor<4xf32>, %arg2 : memref<4xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
 func @simple_transpose_entry_dispatch_0(%arg0: memref<12x12xf32>, %arg1: memref<12x12xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL1:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1LOADPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[GLOBALIDX]], [[GLOBALIDY]]{{\]}}
    // CHECK: [[VAL2:%.*]] = spv.Load "StorageBuffer" [[ARG1LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x12xf32>) : tensor<12x12xf32>
    %1 = "xla_hlo.transpose"(%0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<12x12xf32>) -> tensor<12x12xf32>
    // CHECK: [[RESULT:%.*]] = spv.FAdd [[VAL1]], [[VAL2]]
    %2 = "xla_hlo.add"(%0, %1) : (tensor<12x12xf32>, tensor<12x12xf32>) -> tensor<12x12xf32>
    iree.store_output(%2 : tensor<12x12xf32>, %arg1 : memref<12x12xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    // expected-error @+1 {{unable to map from launch id to element to compute within a workitem}}
    iree.store_output(%0 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42]> : tensor<1xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    // expected-error @+1 {{unable to map from launch id to element to compute within a workitem}}
    iree.store_output(%0 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<42xi32>, %arg1: memref<42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<42xi32>) : tensor<42xi32>
    // expected-error @+1 {{unable to map from launch id to element to compute within a workitem}}
    iree.store_output(%0 : tensor<42xi32>, %arg1 : memref<42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  // CHECK: func [[FN:@simple_load_store_entry_dispatch_0]]()
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42x1xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[GLOBALIDX]], [[ZERO2]]{{\]}}
    %0 = iree.load_input(%arg0 : memref<12x42x1xi32>) : tensor<12x42x1xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<12x42x1xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  // CHECK: func [[FN:@simple_load_store_entry_dispatch_0]]()
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42x1x1xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[GLOBALIDX]], [[ZERO2]], [[ZERO2]]{{\]}}
    %0 = iree.load_input(%arg0 : memref<12x42x1x1xi32>) : tensor<12x42x1x1xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<12x42x1x1xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  // CHECK: func [[FN:@simple_load_store_entry_dispatch_0]]()
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xi32>, %arg1: memref<12x42x1x1xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<12x42xi32>) -> tensor<12x42x1x1xi32>
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ZERO3:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[GLOBALIDY]], [[GLOBALIDX]], [[ZERO3]], [[ZERO3]]{{\]}}
    iree.store_output(%1 : tensor<12x42x1x1xi32>, %arg1 : memref<12x42x1x1xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  // CHECK: func [[FN:@simple_load_store_entry_dispatch_0]]()
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xi32>, %arg1: memref<12x1x1x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    %1 = "xla_hlo.reshape"(%0) : (tensor<12x42xi32>) -> tensor<12x1x1x42xi32>
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ZERO3:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[GLOBALIDY]], [[ZERO3]], [[ZERO3]], [[GLOBALIDX]]{{\]}}
    iree.store_output(%1 : tensor<12x1x1x42xi32>, %arg1 : memref<12x1x1x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  // CHECK: func [[FN:@simple_load_store_entry_dispatch_0]]()
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x1x1x42xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[ZERO2]], [[ZERO2]], [[GLOBALIDX]]{{\]}}
    %0 = iree.load_input(%arg0 : memref<12x1x1x42xi32>) : tensor<12x1x1x42xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<12x1x1x42xi32>) -> tensor<12x42xi32>
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO3:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO3]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  // CHECK: func [[FN:@simple_load_store_entry_dispatch_0]]()
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xi32>, %arg1: memref<3x12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 3]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[GLOBALIDZ:%.*]] = spv.CompositeExtract [[GLOBALID]][2 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<12x42xi32>) -> tensor<3x12x42xi32>
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[GLOBALIDZ]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<3x12x42xi32>, %arg1 : memref<3x12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  // CHECK: func [[FN:@simple_load_store_entry_dispatch_0]]()
  func @simple_load_store_entry_dispatch_0(%arg0: memref<i32>, %arg1: memref<3x12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 3]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[GLOBALIDZ:%.*]] = spv.CompositeExtract [[GLOBALID]][2 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO]]{{\]}}
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<i32>) : tensor<i32>
    %1 = "xla_hlo.broadcast_in_dim"(%0) : (tensor<i32>) -> tensor<3x12x42xi32>
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO1]], [[GLOBALIDZ]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<3x12x42xi32>, %arg1 : memref<3x12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  // CHECK: func [[FN:@simple_load_store_entry_dispatch_0]]()
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xi32>, %arg1: memref<3x12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 3]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[GLOBALIDZ:%.*]] = spv.CompositeExtract [[GLOBALID]][2 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = "xla_hlo.broadcast"(%0) {broadcast_sizes = dense<[3]> : tensor<1xi64>} : (tensor<12x42xi32>) -> tensor<3x12x42xi32>
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[GLOBALIDZ]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<3x12x42xi32>, %arg1 : memref<3x12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK:spv.module "Logical" "GLSL450"
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  // CHECK: func [[FN:@simple_load_store_entry_dispatch_0]]()
  func @simple_load_store_entry_dispatch_0(%arg0: memref<i32>, %arg1: memref<3x12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 3]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[GLOBALIDZ:%.*]] = spv.CompositeExtract [[GLOBALID]][2 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO]]{{\]}}
    // CHECK: [[VAL:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
    %0 = iree.load_input(%arg0 : memref<i32>) : tensor<i32>
    %1 = "xla_hlo.broadcast"(%0) {broadcast_sizes = dense<[3, 12, 42]>: tensor<3xi64>} : (tensor<i32>) -> tensor<3x12x42xi32>
    // CHECK: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK: [[ARG1STOREPTR:%.*]] = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO1]], [[GLOBALIDZ]], [[GLOBALIDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: spv.Store "StorageBuffer" [[ARG1STOREPTR]], [[VAL]]
    iree.store_output(%1 : tensor<3x12x42xi32>, %arg1 : memref<3x12x42xi32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FOrdGreaterThanEqual {{%.*}}, {{%.*}}
    %2 = cmpf "oge", %0, %1 : tensor<12x42xf32>
    //CHECK: {{%.*}} = spv.Select [[COMPARE]], {{%.*}}, {{%.*}}
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FOrdEqual {{%.*}}, {{%.*}}
    %2 = cmpf "oeq", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FOrdGreaterThan {{%.*}}, {{%.*}}
    %2 = cmpf "ogt", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FOrdLessThan {{%.*}}, {{%.*}}
    %2 = cmpf "olt", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FOrdLessThanEqual {{%.*}}, {{%.*}}
    %2 = cmpf "ole", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FOrdNotEqual {{%.*}}, {{%.*}}
    %2 = cmpf "one", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FUnordEqual {{%.*}}, {{%.*}}
    %2 = cmpf "ueq", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FUnordGreaterThanEqual {{%.*}}, {{%.*}}
    %2 = cmpf "uge", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FUnordGreaterThan {{%.*}}, {{%.*}}
    %2 = cmpf "ugt", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FUnordLessThan {{%.*}}, {{%.*}}
    %2 = cmpf "ult", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FUnordLessThanEqual {{%.*}}, {{%.*}}
    %2 = cmpf "ule", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @simple_load_store_entry_dispatch_0(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.FUnordNotEqual {{%.*}}, {{%.*}}
    %2 = cmpf "une", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @const_float_splat(%arg0: memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: spv.constant 1.000000e+00 : f32
    %0 = constant dense<1.0> : tensor<12xf32>
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0]> : tensor<1xi64>} : (tensor<12xf32>) -> tensor<12x42xf32>
    iree.store_output(%1 : tensor<12x42xf32>, %arg0 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @const_int_splat(%arg0: memref<12x42xi64>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: spv.constant 42 : i64
    %0 = constant dense<42> : tensor<12xi64>
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0]> : tensor<1xi64>} : (tensor<12xi64>) -> tensor<12x42xi64>
    iree.store_output(%1 : tensor<12x42xi64>, %arg0 : memref<12x42xi64>)
    iree.return
  }
}

// -----

module {
  func @const_int_splat(%arg0: memref<2x12x42xi64>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 2]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // expected-error @+1{{unhandled constant lowering unless value is a splat dense element attribute}}
    %0 = constant dense<[42, 21]> : tensor<2xi64>
    %1 = "xla_hlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<[0]> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<2x12x42xi64>
    iree.store_output(%1 : tensor<2x12x42xi64>, %arg0 : memref<2x12x42xi64>)
    iree.return
  }
}

// -----

module {
  func @max(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2 : memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.GLSL.FMax [[VAL1:%.*]], [[VAL2:%.*]] : f32
    %2 = xla_hlo.max %0, %1 : tensor<12x42xf32>
    iree.store_output(%2 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @max(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>, %arg2 : memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = iree.load_input(%arg1 : memref<12x42xi32>) : tensor<12x42xi32>
    //CHECK: [[COMPARE:%.*]] = spv.GLSL.SMax [[VAL1:%.*]], [[VAL2:%.*]] : i32
    %2 = xla_hlo.max %0, %1 : tensor<12x42xi32>
    iree.store_output(%2 : tensor<12x42xi32>, %arg2 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<24 x !spv.array<21 x i32 [4]> [84]> [0]>, StorageBuffer>
  func @simple_load_store_entry_dispatch_0(%arg0: memref<24x21xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK-DAG: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK-DAG: [[ZERO1:%.*]] = spv.constant 0 : i32
    // CHECK-DAG: [[STRIDE0:%.*]] = spv.constant 42 : i32
    // CHECK-DAG: [[L1:%.*]] = spv.IMul [[GLOBALIDY]], [[STRIDE0]] : i32
    // CHECK-DAG: [[L2:%.*]] = spv.IAdd [[L1]], [[GLOBALIDX]] : i32
    // CHECK-DAG: [[STRIDE2:%.*]] = spv.constant 21 : i32
    // CHECK-DAG: [[INDEX0:%.*]] = spv.SDiv [[L2]], [[STRIDE2]] : i32
    // CHECK-DAG: [[INDEX1:%.*]] = spv.SMod [[L2]], [[STRIDE2]] : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[INDEX0]], [[INDEX1]]{{\]}}
    %0 = iree.load_input(%arg0 : memref<24x21xi32>) : tensor<24x21xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<24x21xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<4 x !spv.array<6 x !spv.array<21 x i32 [4]> [84]> [504]> [0]>, StorageBuffer>
  func @simple_load_store_entry_dispatch_0(%arg0: memref<4x6x21xi32>, %arg1: memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK-DAG: [[STRIDE1:%.*]] = spv.constant 126 : i32
    // CHECK-DAG: [[I0:%.*]] = spv.SDiv [[L1:%.*]], [[STRIDE2]] : i32
    // CHECK-DAG: [[I1:%.*]] = spv.SMod [[L1]], [[STRIDE2]] : i32
    // CHECK-DAG: [[STRIDE2:%.*]] = spv.constant 21 : i32
    // CHECK-DAG: [[I2:%.*]] = spv.SDiv [[I1]], [[STRIDE2]] : i32
    // CHECK-DAG: [[I3:%.*]] = spv.SMod [[I1]], [[STRIDE2]] : i32
    // CHECK-DAG: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK-DAG: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO1]], [[I0]], [[I2]], [[I3]]{{\]}}
    %0 = iree.load_input(%arg0 : memref<4x6x21xi32>) : tensor<4x6x21xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<4x6x21xi32>) -> tensor<12x42xi32>
    iree.store_output(%1 : tensor<12x42xi32>, %arg1 : memref<12x42xi32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<24 x !spv.array<21 x i32 [4]> [84]> [0]>, StorageBuffer>
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<12 x !spv.array<6 x !spv.array<7 x i32 [4]> [28]> [168]> [0]>, StorageBuffer>
  func @simple_load_store_entry_dispatch_0(%arg0: memref<24x21xi32>, %arg1: memref<12x6x7xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK-DAG: [[ZERO:%.*]] = spv.constant 0 : i32
    // CHECK-DAG: [[FORTYTWO:%.*]] = spv.constant 42 : i32
    // CHECK-DAG: [[I1:%.*]] = spv.IMul [[GLOBALIDY]], [[FORTYTWO]] : i32
    // CHECK-DAG: [[I2:%.*]] = spv.IAdd [[I1]], [[GLOBALIDX]] : i32
    // CHECK-DAG: [[I3:%.*]] = spv.SDiv [[I2]], [[FORTYTWO]] : i32
    // CHECK-DAG: [[I4:%.*]] = spv.IMul [[I3]], [[FORTYTWO]] : i32
    // CHECK-DAG: [[I5:%.*]] = spv.SMod [[I2]], [[FORTYTWO]] : i32
    // CHECK: [[SEVEN:%.*]] = spv.constant 7 : i32
    // CHECK-DAG: [[I6:%.*]] = spv.SDiv [[I5]], [[SEVEN]] : i32
    // CHECK-DAG: [[I7:%.*]] = spv.IMul [[I6]], [[SEVEN]] : i32
    // CHECK-DAG: [[I8:%.*]] = spv.IAdd [[I4]], [[I7]] : i32
    // CHECK-DAG: [[I9:%.*]] = spv.SMod [[I5]], [[SEVEN]] : i32
    // CHECK-DAG: [[I10:%.*]] = spv.IAdd [[I8]], [[I9]] : i32
    // CHECK: [[TWENTYONE:%.*]] = spv.constant 21 : i32
    // CHECK-DAG: [[I11:%.*]] = spv.SDiv [[I10]], [[TWENTYONE]] : i32
    // CHECK-DAG: [[I12:%.*]] = spv.SMod [[I10]], [[TWENTYONE]] : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO]], [[I11]], [[I12]]
    // CHECK-DAG: [[ARG1PTR:%.*]] = spv._address_of [[ARG1VAR]]
    // CHECK-DAG: [[ZERO2:%.*]] = spv.constant 0 : i32
    // CHECK: {{%.*}} = spv.AccessChain [[ARG1PTR]]{{\[}}[[ZERO2]], [[I3]], [[I6]], [[I9]]{{\]}}
    %0 = iree.load_input(%arg0 : memref<24x21xi32>) : tensor<24x21xi32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<24x21xi32>) -> tensor<12x6x7xi32>
    iree.store_output(%1 : tensor<12x6x7xi32>, %arg1 : memref<12x6x7xi32>)
    iree.return
  }
}

// -----

module {
  func @exp(%arg0: memref<12x42xf32>, %arg2 : memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: {{%.*}} = spv.GLSL.Exp {{%.*}} : f32
    %2 = "xla_hlo.exp"(%0) : (tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%2 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @and(%arg0: memref<12x42xi1>, %arg1: memref<12x42xi1>, %arg2: memref<12x42xi1>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.ordinal = 0 : i1} {
    %0 = iree.load_input(%arg0 : memref<12x42xi1>) : tensor<12x42xi1>
    %1 = iree.load_input(%arg1 : memref<12x42xi1>) : tensor<12x42xi1>
    //CHECK: {{%.*}} = spv.LogicalAnd {{%.*}}, {{%.*}} : i1
    %2 = xla_hlo.and %0, %1 : tensor<12x42xi1>
    iree.store_output(%2 : tensor<12x42xi1>, %arg2 : memref<12x42xi1>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  func @reverse_2d(%arg0: memref<12x12xf32>, %arg1 : memref<12x12xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[12, 12]> : tensor<2xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO:%.*]] = spv.constant 0 : i32
    // CHECK: [[NEGATIVE_ONE:%.*]] = spv.constant -1 : i32
    // CHECK: [[NEGATIVE_IDY:%.*]] = spv.IMul [[GLOBALIDY]], [[NEGATIVE_ONE]] : i32
    // CHECK: [[ELEVEN:%.*]] = spv.constant 11 : i32
    // CHECK: [[REVERSE_IDY:%.*]] = spv.IAdd [[NEGATIVE_IDY]], [[ELEVEN]] : i32
    // CHECK: [[NEGATIVE_IDX:%.*]] = spv.IMul [[GLOBALIDX]], [[NEGATIVE_ONE]] : i32
    // CHECK: [[REVERSE_IDX:%.*]] = spv.IAdd [[NEGATIVE_IDX]], [[ELEVEN]] : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO]], [[REVERSE_IDY]], [[REVERSE_IDX]]{{\]}}
    // CHECK: [[VAL0:%.*]]  = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<12x12xf32>) : tensor<12x12xf32>
    %1 = "xla_hlo.reverse"(%0) {dimensions = dense<[1, 0]> : tensor<2xi64>} : (tensor<12x12xf32>) -> tensor<12x12xf32>
    iree.store_output(%1 : tensor<12x12xf32>, %arg1 : memref<12x12xf32>)
    iree.return
  }
}

// -----

module {
  // CHECK: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  // CHECK: spv.globalVariable [[ARG0VAR:@.*]] bind(0, 0)
  // CHECK: spv.globalVariable [[ARG1VAR:@.*]] bind(0, 1)
  func @reverse_3d(%arg0: memref<3x3x3xf32>, %arg1 : memref<3x3x3xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[3, 3, 3]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%.*]] = spv._address_of [[GLOBALIDVAR]]
    // CHECK: [[GLOBALID:%.*]] = spv.Load "Input" [[GLOBALIDPTR]]
    // CHECK: [[GLOBALIDX:%.*]] = spv.CompositeExtract [[GLOBALID]][0 : i32]
    // CHECK: [[GLOBALIDY:%.*]] = spv.CompositeExtract [[GLOBALID]][1 : i32]
    // CHECK: [[GLOBALIDZ:%.*]] = spv.CompositeExtract [[GLOBALID]][2 : i32]
    // CHECK: [[ARG0PTR:%.*]] = spv._address_of [[ARG0VAR]]
    // CHECK: [[ZERO:%.*]] = spv.constant 0 : i32
    // CHECK: [[NEGATIVE_ONE:%.*]] = spv.constant -1 : i32
    // CHECK: [[NEGATIVE_IDY:%.*]] = spv.IMul [[GLOBALIDY]], [[NEGATIVE_ONE]] : i32
    // CHECK: [[TWO:%.*]] = spv.constant 2 : i32
    // CHECK: [[REVERSE_IDY:%.*]] = spv.IAdd [[NEGATIVE_IDY]], [[TWO]] : i32
    // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0PTR]]{{\[}}[[ZERO]], [[GLOBALIDZ]], [[REVERSE_IDY]], [[GLOBALIDX]]{{\]}}
    // CHECK: [[VAL0:%.*]]  = spv.Load "StorageBuffer" [[ARG0LOADPTR]] : f32
    %0 = iree.load_input(%arg0 : memref<3x3x3xf32>) : tensor<3x3x3xf32>
    %1 = "xla_hlo.reverse"(%0) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    iree.store_output(%1 : tensor<3x3x3xf32>, %arg1 : memref<3x3x3xf32>)
    iree.return
  }
}
