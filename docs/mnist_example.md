# MNIST IR Example

This shows the MNIST MLP model as it is compiled from Keras, lowered to XLA HLO,
and then lowered to an IREE module with SPIR-V. Several steps are omitted for
brevity.

## TensorFlow Keras Model

```python
def simple_mnist_model(input_shape):
  """Creates a simple (multi-layer perceptron) MNIST model."""
  model = tf.keras.models.Sequential()
  # Flatten to a 1d array (e.g. 28x28 -> 784)
  model.add(tf.keras.layers.Flatten(input_shape=input_shape))
  # Fully-connected neural layer with 128 neurons, RELU activation
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  # Fully-connected neural layer returning probability scores for each class
  model.add(tf.keras.layers.Dense(10, activation='softmax'))
  return model
```

## XLA HLO

**NOTE**: this uses placeholder weights to keep the page from being a few
thousand lines of floats.

```mlir
module {
  func @main(%arg0: tensor<1x28x28x1xf32>) -> tuple<tensor<1x10xf32>>
  attributes {iree.module.export} {
    %cst = constant  {name = "constant.9"} dense<0.5> : tensor<f32>
    %0 = "mhlo.broadcast_in_dim"(%cst) {name = "broadcast.10"} : (tensor<f32>) -> tensor<1x128xf32>
    %1 = "mhlo.copy"(%arg0) {name = "copy.1"} : (tensor<1x28x28x1xf32>) -> tensor<1x28x28x1xf32>
    %2 = "mhlo.reshape"(%1) {name = "reshape.2"} : (tensor<1x28x28x1xf32>) -> tensor<1x28x28x1xf32>
    %3 = "mhlo.reshape"(%2) {name = "reshape.3"} : (tensor<1x28x28x1xf32>) -> tensor<1x784xf32>
    %cst_0 = constant  {name = "constant.4"} dense<0.5> : tensor<784x128xf32>
    %4 = "mhlo.dot"(%3, %cst_0) {name = "dot.5", precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x784xf32>, tensor<784x128xf32>) -> tensor<1x128xf32>
    %cst_1 = constant  {name = "constant.6"} dense<0.5> : tensor<128xf32>
    %5 = "mhlo.broadcast_in_dim"(%cst_1) {broadcast_dimensions = dense<1> : tensor<1xi64>, name = "broadcast.7"} : (tensor<128xf32>) -> tensor<1x128xf32>
    %6 = "mhlo.add"(%4, %5) {name = "add.8"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %7 = "mhlo.maximum"(%0, %6) {name = "maximum.11"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %cst_2 = constant  {name = "constant.12"} dense<0.5> : tensor<128x10xf32>
    %8 = "mhlo.dot"(%7, %cst_2) {name = "dot.13", precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128xf32>, tensor<128x10xf32>) -> tensor<1x10xf32>
    %cst_3 = constant  {name = "constant.14"} dense<0.5> : tensor<10xf32>
    %9 = "mhlo.broadcast_in_dim"(%cst_3) {broadcast_dimensions = dense<1> : tensor<1xi64>, name = "broadcast.15"} : (tensor<10xf32>) -> tensor<1x10xf32>
    %10 = "mhlo.add"(%8, %9) {name = "add.16"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %cst_4 = constant  {name = "constant.17"} dense<0xFF800000> : tensor<f32>
    %11 = "mhlo.reduce"(%10, %cst_4) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):   // no predecessors
      %20 = "mhlo.maximum"(%arg1, %arg2) {name = "maximum.21"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%20) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %12 = "mhlo.broadcast_in_dim"(%11) {broadcast_dimensions = dense<0> : tensor<1xi64>, name = "broadcast.23"} : (tensor<1xf32>) -> tensor<1x10xf32>
    %13 = "mhlo.subtract"(%10, %12) {name = "subtract.24"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %14 = "mhlo.exponential"(%13) {name = "exponential.25"} : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %cst_5 = constant  {name = "constant.27"} dense<0.5> : tensor<f32>
    %15 = "mhlo.reduce"(%14, %cst_5) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):   // no predecessors
      %21 = "mhlo.add"(%arg3, %arg4) {name = "add.31"} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%21) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
    %16 = "mhlo.broadcast_in_dim"(%15) {broadcast_dimensions = dense<0> : tensor<1xi64>, name = "broadcast.34"} : (tensor<1xf32>) -> tensor<1x10xf32>
    %17 = "mhlo.divide"(%14, %16) {name = "divide.35"} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %18 = "mhlo.reshape"(%17) {name = "reshape.36"} : (tensor<1x10xf32>) -> tensor<1x10xf32>
    %19 = "mhlo.tuple"(%18) {name = "tuple.37"} : (tensor<1x10xf32>) -> tuple<tensor<1x10xf32>>
    return %19 : tuple<tensor<1x10xf32>>
  }
}
```

## IREE IR (pre-backend lowering)

Here's the lowered, outlined, and compiler-annotated version of the above in the
IREE sequencer dialect.

```mlir
module {
  iree.multi_arch_executable @main_ex_dispatch_0[0]() {
    iree.executable[0](Unspecified) {
      module {
        func @main_entry_dispatch_0(%arg0: memref<1x28x28x1xf32>, %arg1: memref<1x784xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[784, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
          %0 = iree.load_input(%arg0 : memref<1x28x28x1xf32>) : tensor<1x28x28x1xf32>
          %1 = "mhlo.copy"(%0) {name = "copy.1"} : (tensor<1x28x28x1xf32>) -> tensor<1x28x28x1xf32>
          %2 = "mhlo.reshape"(%1) {name = "reshape.3"} : (tensor<1x28x28x1xf32>) -> tensor<1x784xf32>
          iree.store_output(%2 : tensor<1x784xf32>, %arg1 : memref<1x784xf32>)
          iree.return
        }
      }
    }
  }
  iree.multi_arch_executable @main_ex_dispatch_1[1]() {
    iree.executable[1](Unspecified) {
      module {
        func @main_entry_dispatch_1(%arg0: memref<1x784xf32>, %arg1: memref<784x128xf32>, %arg2: memref<1x128xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[128, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
          %0 = iree.load_input(%arg0 : memref<1x784xf32>) : tensor<1x784xf32>
          %1 = iree.load_input(%arg1 : memref<784x128xf32>) : tensor<784x128xf32>
          %2 = "mhlo.dot"(%0, %1) {name = "dot.5", precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x784xf32>, tensor<784x128xf32>) -> tensor<1x128xf32>
          iree.store_output(%2 : tensor<1x128xf32>, %arg2 : memref<1x128xf32>)
          iree.return
        }
      }
    }
  }
  iree.multi_arch_executable @main_ex_dispatch_2[2]() {
    iree.executable[2](Unspecified) {
      module {
        func @main_entry_dispatch_2(%arg0: memref<1x128xf32>, %arg1: memref<1x128xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[128, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
          %0 = iree.load_input(%arg0 : memref<1x128xf32>) : tensor<1x128xf32>
          %cst = constant dense<5.000000e-01> : tensor<128xf32>
          %cst_0 = constant dense<5.000000e-01> : tensor<f32>
          %1 = "mhlo.broadcast_in_dim"(%cst_0) {name = "broadcast.10"} : (tensor<f32>) -> tensor<1x128xf32>
          %2 = "mhlo.broadcast_in_dim"(%cst) {broadcast_dimensions = dense<1> : tensor<1xi64>, name = "broadcast.7"} : (tensor<128xf32>) -> tensor<1x128xf32>
          %3 = addf %0, %2 : tensor<1x128xf32>
          %4 = mhlo.maximum %1, %3 {name = "maximum.11"} : tensor<1x128xf32>
          iree.store_output(%4 : tensor<1x128xf32>, %arg1 : memref<1x128xf32>)
          iree.return
        }
      }
    }
  }
  iree.multi_arch_executable @main_ex_dispatch_3[3]() {
    iree.executable[3](Unspecified) {
      module {
        func @main_entry_dispatch_3(%arg0: memref<1x128xf32>, %arg1: memref<128x10xf32>, %arg2: memref<1x10xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[10, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
          %0 = iree.load_input(%arg0 : memref<1x128xf32>) : tensor<1x128xf32>
          %1 = iree.load_input(%arg1 : memref<128x10xf32>) : tensor<128x10xf32>
          %2 = "mhlo.dot"(%0, %1) {name = "dot.13", precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x128xf32>, tensor<128x10xf32>) -> tensor<1x10xf32>
          iree.store_output(%2 : tensor<1x10xf32>, %arg2 : memref<1x10xf32>)
          iree.return
        }
      }
    }
  }
  iree.multi_arch_executable @main_ex_dispatch_4[4]() {
    iree.executable[4](Unspecified) {
      module {
        func @main_entry_dispatch_4(%arg0: memref<1x10xf32>, %arg1: memref<1x10xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[10, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
          %0 = iree.load_input(%arg0 : memref<1x10xf32>) : tensor<1x10xf32>
          %cst = constant dense<5.000000e-01> : tensor<10xf32>
          %1 = "mhlo.broadcast_in_dim"(%cst) {broadcast_dimensions = dense<1> : tensor<1xi64>, name = "broadcast.15"} : (tensor<10xf32>) -> tensor<1x10xf32>
          %2 = addf %0, %1 : tensor<1x10xf32>
          iree.store_output(%2 : tensor<1x10xf32>, %arg1 : memref<1x10xf32>)
          iree.return
        }
      }
    }
  }
  iree.multi_arch_executable @main_ex_dispatch_5[5]() {
    iree.executable[5](Unspecified) {
      module {
        func @main_entry_dispatch_5(%arg0: memref<1x10xf32>, %arg1: memref<1xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<1> : tensor<3xi32>, iree.ordinal = 0 : i32} {
          %0 = iree.load_input(%arg0 : memref<1x10xf32>) : tensor<1x10xf32>
          %cst = constant dense<0xFF800000> : tensor<f32>
          %1 = "mhlo.reduce"(%0, %cst) ( {
          ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>): // no predecessors
            %2 = mhlo.maximum %arg2, %arg3 {name = "maximum.21"} : tensor<f32>
            "mhlo.return"(%2) : (tensor<f32>) -> ()
          }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
          iree.store_output(%1 : tensor<1xf32>, %arg1 : memref<1xf32>)
          iree.return
        }
      }
    }
  }
  iree.multi_arch_executable @main_ex_dispatch_6[6]() {
    iree.executable[6](Unspecified) {
      module {
        func @main_entry_dispatch_6(%arg0: memref<1x10xf32>, %arg1: memref<1xf32>, %arg2: memref<1x10xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[10, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
          %0 = iree.load_input(%arg0 : memref<1x10xf32>) : tensor<1x10xf32>
          %1 = iree.load_input(%arg1 : memref<1xf32>) : tensor<1xf32>
          %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>, name = "broadcast.23"} : (tensor<1xf32>) -> tensor<1x10xf32>
          %3 = subf %0, %2 : tensor<1x10xf32>
          %4 = "mhlo.exponential"(%3) {name = "exponential.25"} : (tensor<1x10xf32>) -> tensor<1x10xf32>
          iree.store_output(%4 : tensor<1x10xf32>, %arg2 : memref<1x10xf32>)
          iree.return
        }
      }
    }
  }
  iree.multi_arch_executable @main_ex_dispatch_7[7]() {
    iree.executable[7](Unspecified) {
      module {
        func @main_entry_dispatch_7(%arg0: memref<1x10xf32>, %arg1: memref<1xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<1> : tensor<3xi32>, iree.ordinal = 0 : i32} {
          %0 = iree.load_input(%arg0 : memref<1x10xf32>) : tensor<1x10xf32>
          %cst = constant dense<5.000000e-01> : tensor<f32>
          %1 = "mhlo.reduce"(%0, %cst) ( {
          ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>): // no predecessors
            %2 = addf %arg2, %arg3 : tensor<f32>
            "mhlo.return"(%2) : (tensor<f32>) -> ()
          }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
          iree.store_output(%1 : tensor<1xf32>, %arg1 : memref<1xf32>)
          iree.return
        }
      }
    }
  }
  iree.multi_arch_executable @main_ex_dispatch_8[8]() {
    iree.executable[8](Unspecified) {
      module {
        func @main_entry_dispatch_8(%arg0: memref<1xf32>, %arg1: memref<1x10xf32>, %arg2: memref<1x10xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[10, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
          %0 = iree.load_input(%arg0 : memref<1xf32>) : tensor<1xf32>
          %1 = iree.load_input(%arg1 : memref<1x10xf32>) : tensor<1x10xf32>
          %2 = "mhlo.broadcast_in_dim"(%0) {broadcast_dimensions = dense<0> : tensor<1xi64>, name = "broadcast.34"} : (tensor<1xf32>) -> tensor<1x10xf32>
          %3 = divf %1, %2 : tensor<1x10xf32>
          iree.store_output(%3 : tensor<1x10xf32>, %arg2 : memref<1x10xf32>)
          iree.return
        }
      }
    }
  }
  func @main(%arg0: memref<1x28x28x1xf32>) -> memref<1x10xf32>
  attributes  {iree.module.export} {
    %0 = "iree_ll_seq.constant"() {value = dense<5.000000e-01> : tensor<784x128xf32>} : () -> memref<784x128xf32>
    %1 = "iree_ll_seq.constant"() {value = dense<5.000000e-01> : tensor<128x10xf32>} : () -> memref<128x10xf32>
    %2 = "iree_ll_seq.alloc_heap"() : () -> memref<1x784xf32>
    iree_ll_seq.static_dispatch main_ex_dispatch_0::main_entry_dispatch_0[dense<[784, 1, 1]> : tensor<3xi32>](%arg0, %2) : (memref<1x28x28x1xf32>, memref<1x784xf32>) -> ()
    %3 = "iree_ll_seq.alloc_heap"() : () -> memref<1x128xf32>
    iree_ll_seq.static_dispatch main_ex_dispatch_1::main_entry_dispatch_1[dense<[128, 1, 1]> : tensor<3xi32>](%2, %0, %3) : (memref<1x784xf32>, memref<784x128xf32>, memref<1x128xf32>) -> ()
    %4 = "iree_ll_seq.alloc_heap"() : () -> memref<1x128xf32>
    iree_ll_seq.static_dispatch main_ex_dispatch_2::main_entry_dispatch_2[dense<[128, 1, 1]> : tensor<3xi32>](%3, %4) : (memref<1x128xf32>, memref<1x128xf32>) -> ()
    %5 = "iree_ll_seq.alloc_heap"() : () -> memref<1x10xf32>
    iree_ll_seq.static_dispatch main_ex_dispatch_3::main_entry_dispatch_3[dense<[10, 1, 1]> : tensor<3xi32>](%4, %1, %5) : (memref<1x128xf32>, memref<128x10xf32>, memref<1x10xf32>) -> ()
    %6 = "iree_ll_seq.alloc_heap"() : () -> memref<1x10xf32>
    iree_ll_seq.static_dispatch main_ex_dispatch_4::main_entry_dispatch_4[dense<[10, 1, 1]> : tensor<3xi32>](%5, %6) : (memref<1x10xf32>, memref<1x10xf32>) -> ()
    %7 = "iree_ll_seq.alloc_heap"() : () -> memref<1xf32>
    iree_ll_seq.static_dispatch main_ex_dispatch_5::main_entry_dispatch_5[dense<1> : tensor<3xi32>](%6, %7) : (memref<1x10xf32>, memref<1xf32>) -> ()
    %8 = "iree_ll_seq.alloc_heap"() : () -> memref<1x10xf32>
    iree_ll_seq.static_dispatch main_ex_dispatch_6::main_entry_dispatch_6[dense<[10, 1, 1]> : tensor<3xi32>](%6, %7, %8) : (memref<1x10xf32>, memref<1xf32>, memref<1x10xf32>) -> ()
    %9 = "iree_ll_seq.alloc_heap"() : () -> memref<1xf32>
    iree_ll_seq.static_dispatch main_ex_dispatch_7::main_entry_dispatch_7[dense<1> : tensor<3xi32>](%8, %9) : (memref<1x10xf32>, memref<1xf32>) -> ()
    %10 = "iree_ll_seq.alloc_heap"() : () -> memref<1x10xf32>
    iree_ll_seq.static_dispatch main_ex_dispatch_8::main_entry_dispatch_8[dense<[10, 1, 1]> : tensor<3xi32>](%9, %8, %10) : (memref<1xf32>, memref<1x10xf32>, memref<1x10xf32>) -> ()
    iree_ll_seq.return %10 : memref<1x10xf32>
  }
}
```

**NOTE**: this is effectively compiling in -O0, which is why the buffers are not
aliased and some dispatch region fusing is not performed. As we get things going
we'll be adding simple optimizations that can operate on this IR to elide almost
all copies and externalize allocations to transient pooled memory.

## Final IREE Module with SPIR-V

TODO(benvanik): once reductions are done.
