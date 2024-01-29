// Sample spec that matches an MLP example and forwards to 
// an implementation implemented by a system plugin.
// Is used along with samples/custom_dispatch/cpu/plugin/mlp.mlir

#x86_64_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 32 : index,
  target_triple = "x86_64-none-elf"
}>

#cpu_target = #hal.device.target<"llvm-cpu", {
  executable_targets = [
    #x86_64_target
  ]
}>

module attributes {transform.with_named_sequence} {

  // Executable that stages call to the external functions.
  hal.executable private @executable {
    hal.executable.variant public @x86_64 target(#x86_64_target) {
      hal.executable.export public @mlp ordinal(0)
          layout(#hal.pipeline.layout<push_constants = 3, sets = [
            <0, bindings = [
              <0, storage_buffer, ReadOnly>,
              <1, storage_buffer, ReadOnly>,
              <2, storage_buffer>
            ]>
          ]>) {
      ^bb0(%device : !hal.device):
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func private @mlp_external(%lhs : memref<?x?xf32>, %rhs : memref<?x?xf32>, %result : memref<?x?xf32>, %m : i32, %n : i32, %k : i32)
        func.func @mlp() {
          %m_i32 = hal.interface.constant.load[0] : i32
          %n_i32 = hal.interface.constant.load[1] : i32
          %k_i32 = hal.interface.constant.load[2] : i32
          %c0 = arith.constant 0 : index
          %m = arith.index_cast %m_i32 : i32 to index
          %n = arith.index_cast %n_i32 : i32 to index
          %k = arith.index_cast %k_i32 : i32 to index
          %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<?x?xf32>{%m, %k}
          %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<?x?xf32>{%k, %n}
          %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<?x?xf32>{%m, %n}
          func.call @mlp_external(%lhs, %rhs, %result, %m_i32, %n_i32, %k_i32) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>, i32, i32, i32) -> ()
          return
        }
      }
    }
  }

  func.func private @call_mlp(%lhs : tensor<?x?xf32>, %rhs : tensor<?x?xf32>, %init1 : tensor<?x?xf32>, %init2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %n = tensor.dim %rhs, %c1 : tensor<?x?xf32>
    %k = tensor.dim %lhs, %c1 : tensor<?x?xf32>
    %m_i32 = arith.index_cast %m : index to i32
    %n_i32 = arith.index_cast %n : index to i32
    %k_i32 = arith.index_cast %k : index to i32

    %mlp_result = flow.dispatch @executable::@x86_64::@mlp[](%lhs, %rhs, %m_i32, %n_i32, %k_i32) {
      hal.interface.bindings = [
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ],
      // HACK: keep the executable live through DCE. Only required when
      // using the automatic variant selection.
      hal.executable.ref = [@executable]
    } : (tensor<?x?xf32>{%m, %k}, tensor<?x?xf32>{%k, %n}, i32, i32, i32) -> tensor<?x?xf32>{%m, %n}  
    return %mlp_result : tensor<?x?xf32>    
  }

  transform.named_sequence @match_mlp(%root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %init1 : tensor<?x?xf32>, %init2 : tensor<?x?xf32>):
        %cst = arith.constant 0.0 : f32
        %fill = linalg.fill ins(%cst : f32) outs(%init1 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %matmul = linalg.matmul
            ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
                outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
        %relu = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                             affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]}
            ins(%matmul : tensor<?x?xf32>)
            outs(%init2 : tensor<?x?xf32>) {
          ^bb0(%b0 : f32, %b1 : f32):
            %0 = arith.maximumf %b0, %cst : f32
            linalg.yield %0 : f32
          } -> tensor<?x?xf32>
      } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }


  // Rewrite callback for `transform.foreach_match`. The input signature for
  // this sequence must match exactly with the outputs of the matcher. In this
  // case the matcher returns the inputs and outputs to the matched dag directly
  // so we just insert a call to the hand authored function above.
  transform.named_sequence @cast_and_call_dag(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.iree.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.iree.import_symbol @executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.iree.import_symbol @call_mlp into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.func.cast_and_call %func(%ins) -> %out after %root {
      // This specifies how to resolve type mismatches between the arguments
      // of the function and the inputs from the matcher. In this example,
      // the only casts this will generate are same-rank tensor casts that
      // drop static information.
      transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  // Entry point for the transform interpreter, nested on the full module. This
  // is because the rewrites needed for importing the custom kernel needs to
  // add a new symbol to the module's symbol table.
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    // Gather the set of functions within the module.
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op   
    // For each function in the module, run the matcher on all contained
    // operations.
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
          // <matcher name> -> <rewriter name>
          // Multiple matcher-action pairs can be specified comma separated,
          // here we are only doing a single kind of match and replace.
          @match_mlp -> @cast_and_call_dag
        : (!transform.any_op) -> (!transform.any_op)
    }
    // Cleanup leftover dead code; cast_and_call does not do replacement, only
    // rewires uses.
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}
