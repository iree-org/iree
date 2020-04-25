// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  func @select_ford_ge(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_ford_ge"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FOrdGreaterThanEqual %{{.+}}, %{{.+}}
    %2 = cmpf "oge", %0, %1 : tensor<12x42xf32>
    //CHECK: %{{.+}} = spv.Select %[[COMPARE]], %{{.+}}, %{{.+}}
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_ford_eq(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_ford_eq"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FOrdEqual %{{.+}}, %{{.+}}
    %2 = cmpf "oeq", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_ford_gt(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_ford_gt"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FOrdGreaterThan %{{.+}}, %{{.+}}
    %2 = cmpf "ogt", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_ford_lt(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_ford_lt"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FOrdLessThan %{{.+}}, %{{.+}}
    %2 = cmpf "olt", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_ford_le(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_ford_le"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FOrdLessThanEqual %{{.+}}, %{{.+}}
    %2 = cmpf "ole", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_ford_ne(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_ford_ne"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FOrdNotEqual %{{.+}}, %{{.+}}
    %2 = cmpf "one", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_funord_eq(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_funord_eq"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FUnordEqual %{{.+}}, %{{.+}}
    %2 = cmpf "ueq", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_funord_ge(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_funord_ge"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FUnordGreaterThanEqual %{{.+}}, %{{.+}}
    %2 = cmpf "uge", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_funord_gt(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_funord_gt"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FUnordGreaterThan %{{.+}}, %{{.+}}
    %2 = cmpf "ugt", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_funord_lt(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_funord_lt"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FUnordLessThan %{{.+}}, %{{.+}}
    %2 = cmpf "ult", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_funord_le(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_funord_le"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FUnordLessThanEqual %{{.+}}, %{{.+}}
    %2 = cmpf "ule", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_funord_ne(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2: memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "select_funord_ne"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %[[COMPARE:.+]] = spv.FUnordNotEqual %{{.+}}, %{{.+}}
    %2 = cmpf "une", %0, %1 : tensor<12x42xf32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xf32>, tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%3 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @select_int_eq(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>, %arg2: memref<12x42xi32>)
  attributes {iree.dispatch_fn_name = "select_int_eq"} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = iree.load_input(%arg1 : memref<12x42xi32>) : tensor<12x42xi32>
    //CHECK: %[[COMPARE:.+]] = spv.IEqual %{{.+}}, %{{.+}}
    %2 = cmpi "eq", %0, %1 : tensor<12x42xi32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xi32>, tensor<12x42xi32>) -> tensor<12x42xi32>
    iree.store_output(%3 : tensor<12x42xi32>, %arg2 : memref<12x42xi32>)
    return
  }
}

// -----

module {
  func @select_int_ne(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>, %arg2: memref<12x42xi32>)
  attributes {iree.dispatch_fn_name = "select_int_ne"} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = iree.load_input(%arg1 : memref<12x42xi32>) : tensor<12x42xi32>
    //CHECK: %[[COMPARE:.+]] = spv.INotEqual %{{.+}}, %{{.+}}
    %2 = cmpi "ne", %0, %1 : tensor<12x42xi32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xi32>, tensor<12x42xi32>) -> tensor<12x42xi32>
    iree.store_output(%3 : tensor<12x42xi32>, %arg2 : memref<12x42xi32>)
    return
  }
}

// -----

module {
  func @select_int_lt(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>, %arg2: memref<12x42xi32>)
  attributes {iree.dispatch_fn_name = "select_int_lt"} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = iree.load_input(%arg1 : memref<12x42xi32>) : tensor<12x42xi32>
    //CHECK: %[[COMPARE:.+]] = spv.SLessThan %{{.+}}, %{{.+}}
    %2 = cmpi "slt", %0, %1 : tensor<12x42xi32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xi32>, tensor<12x42xi32>) -> tensor<12x42xi32>
    iree.store_output(%3 : tensor<12x42xi32>, %arg2 : memref<12x42xi32>)
    return
  }
}

// -----

module {
  func @select_int_le(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>, %arg2: memref<12x42xi32>)
  attributes {iree.dispatch_fn_name = "select_int_le"} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = iree.load_input(%arg1 : memref<12x42xi32>) : tensor<12x42xi32>
    //CHECK: %[[COMPARE:.+]] = spv.SLessThanEqual %{{.+}}, %{{.+}}
    %2 = cmpi "sle", %0, %1 : tensor<12x42xi32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xi32>, tensor<12x42xi32>) -> tensor<12x42xi32>
    iree.store_output(%3 : tensor<12x42xi32>, %arg2 : memref<12x42xi32>)
    return
  }
}

// -----

module {
  func @select_int_ge(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>, %arg2: memref<12x42xi32>)
  attributes {iree.dispatch_fn_name = "select_int_ge"} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = iree.load_input(%arg1 : memref<12x42xi32>) : tensor<12x42xi32>
    //CHECK: %[[COMPARE:.+]] = spv.SGreaterThanEqual %{{.+}}, %{{.+}}
    %2 = cmpi "sge", %0, %1 : tensor<12x42xi32>
    //CHECK: %{{.+}} = spv.Select %[[COMPARE]], %{{.+}}, %{{.+}}
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xi32>, tensor<12x42xi32>) -> tensor<12x42xi32>
    iree.store_output(%3 : tensor<12x42xi32>, %arg2 : memref<12x42xi32>)
    return
  }
}

// -----

module {
  func @select_int_gt(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>, %arg2: memref<12x42xi32>)
  attributes {iree.dispatch_fn_name = "select_int_gt"} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = iree.load_input(%arg1 : memref<12x42xi32>) : tensor<12x42xi32>
    //CHECK: %[[COMPARE:.+]] = spv.SGreaterThan %{{.+}}, %{{.+}}
    %2 = cmpi "sgt", %0, %1 : tensor<12x42xi32>
    %3 = "xla_hlo.select"(%2, %0, %1) : (tensor<12x42xi1>, tensor<12x42xi32>, tensor<12x42xi32>) -> tensor<12x42xi32>
    iree.store_output(%3 : tensor<12x42xi32>, %arg2 : memref<12x42xi32>)
    return
  }
}
