// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt %s --iree-hal-serialize-all-executables="dump-binaries-path=%t" --iree-metal-compile-to-metallib=false -o %t/output.mlir
// RUN: FileCheck %s --input-file=%t/module_loop_accesses_metal0.metal
// RUN: FileCheck %s --check-prefix=PURE --input-file=%t/module_loop_pure_calls_metal0.metal
// RUN: FileCheck %s --check-prefix=CALL --input-file=%t/module_loop_calls_metal0.metal
// RUN: FileCheck %s --check-prefix=POINTER --input-file=%t/module_loop_pointer_metal0.metal
// RUN: FileCheck %s --check-prefix=ARRAY --input-file=%t/module_loop_array_copy_metal0.metal
// RUN: FileCheck %s --check-prefix=VECTOR --input-file=%t/module_loop_vector_metal0.metal
// RUN: iree-opt %s --iree-hal-serialize-all-executables -o %t/compiled.mlir

// Loop-carried updates need volatile device accesses. Input-only and output-only
// loop accesses, and read/write accesses outside a loop, retain ordinary memory.
// CHECK-NOT: #pragma clang optimize off
// CHECK-LABEL: struct spvDescriptorSetBuffer0
// CHECK-NEXT: {
// CHECK-NEXT: {{    device [^*]+\* _input}} {{\[\[id\(}}0{{\)\]\]}};
// CHECK-NEXT: {{    device [^*]+\* write_only}} {{\[\[id\(}}1{{\)\]\]}};
// CHECK-NEXT: {{    volatile device [^*]+\* read_write}} {{\[\[id\(}}2{{\)\]\]}};
// CHECK-NEXT: {{    device [^*]+\* outside_loop}} {{\[\[id\(}}3{{\)\]\]}};
// CHECK-NEXT: {{    volatile device [^*]+\* copy_read_write}} {{\[\[id\(}}4{{\)\]\]}};
// CHECK-NEXT: {{    volatile device [^*]+\* cast_read_write}} {{\[\[id\(}}5{{\)\]\]}};
// CHECK-NEXT: };
// CHECK-NOT: volatile
// CHECK: kernel void main0

// Calls inside a loop may read and update device memory through their callees.
// CALL: #pragma clang optimize off
// CALL-LABEL: struct spvDescriptorSetBuffer0
// CALL-NEXT: {
// CALL-NEXT: {{    device [^*]+\* _input}} {{\[\[id\(}}0{{\)\]\]}};
// CALL-NEXT: {{    device [^*]+\* read_write}} {{\[\[id\(}}1{{\)\]\]}};
// CALL-NEXT: };
// CALL-NOT: volatile

// A helper with no memory effects retains normal optimisation and pointers.
// PURE-NOT: #pragma clang optimize off
// PURE-LABEL: struct spvDescriptorSetBuffer0
// PURE-NOT: volatile
// PURE: kernel void main0

// A selected pointer has no directly resolvable global access chain.
// POINTER: #pragma clang optimize off
// POINTER-LABEL: struct spvDescriptorSetBuffer0
// POINTER-NEXT: {
// POINTER-NEXT: {{    device [^*]+\* read_write}} {{\[\[id\(}}0{{\)\]\]}};
// POINTER-NEXT: };
// POINTER-NOT: volatile

// Native vector values retain typed volatile access.
// VECTOR-NOT: #pragma clang optimize off
// VECTOR-LABEL: struct spvDescriptorSetBuffer0
// VECTOR: volatile device

// Whole-array copies require ordinary composite assignment operators.
// ARRAY: #pragma clang optimize off
// ARRAY-NOT: volatile
// ARRAY-LABEL: kernel void main0

!vector_buffer = !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
!buffer = !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
!element = !spirv.ptr<i32, StorageBuffer>
!array_buffer = !spirv.ptr<!spirv.struct<(!spirv.array<4xf32, stride=4> [0])>, StorageBuffer>

module {
  hal.executable public @loop_accesses {
    hal.executable.variant public @metal target(<"metal-spirv", "metal-msl-fb">) {
      hal.executable.export public @main0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
      builtin.module {
        spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
          spirv.GlobalVariable @input bind(0, 0) : !buffer
          spirv.GlobalVariable @write_only bind(0, 1) : !buffer
          spirv.GlobalVariable @read_write bind(0, 2) : !buffer
          spirv.GlobalVariable @outside_loop bind(0, 3) : !buffer
          spirv.GlobalVariable @copy_read_write bind(0, 4) : !buffer
          spirv.GlobalVariable @cast_read_write bind(0, 5) : !buffer
          spirv.func @main0() "None" {
            %zero = spirv.Constant 0 : i32
            %one = spirv.Constant 1 : i32
            %limit = spirv.Constant 11 : i32
            %float_one = spirv.Constant 1.0 : f32
            %input = spirv.mlir.addressof @input : !buffer
            %write = spirv.mlir.addressof @write_only : !buffer
            %update = spirv.mlir.addressof @read_write : !buffer
            %outside = spirv.mlir.addressof @outside_loop : !buffer
            %copy = spirv.mlir.addressof @copy_read_write : !buffer
            %cast = spirv.mlir.addressof @cast_read_write : !buffer
            %outside_element = spirv.AccessChain %outside[%zero, %zero] : !buffer, i32, i32 -> !element
            %original = spirv.Load "StorageBuffer" %outside_element : i32
            %next = spirv.IAdd %original, %one : i32
            spirv.Store "StorageBuffer" %outside_element, %next : i32
            spirv.mlir.loop {
              spirv.Branch ^header(%zero : i32)
            ^header(%i: i32):
              %more = spirv.SLessThan %i, %limit : i32
              spirv.BranchConditional %more, ^body, ^exit
            ^body:
              %input_element = spirv.AccessChain %input[%zero, %i] : !buffer, i32, i32 -> !element
              %value = spirv.Load "StorageBuffer" %input_element : i32
              %write_element = spirv.AccessChain %write[%zero, %i] : !buffer, i32, i32 -> !element
              spirv.Store "StorageBuffer" %write_element, %value : i32
              %update_element = spirv.AccessChain %update[%zero, %zero] : !buffer, i32, i32 -> !element
              %old = spirv.Load "StorageBuffer" %update_element : i32
              %sum = spirv.IAdd %old, %value : i32
              spirv.Store "StorageBuffer" %update_element, %sum : i32
              %copy_element = spirv.AccessChain %copy[%zero, %zero] : !buffer, i32, i32 -> !element
              spirv.CopyMemory "StorageBuffer" %copy_element, "StorageBuffer" %input_element : i32
              spirv.CopyMemory "StorageBuffer" %write_element, "StorageBuffer" %copy_element : i32
              %cast_element = spirv.AccessChain %cast[%zero, %zero] : !buffer, i32, i32 -> !element
              %old_bits = spirv.Load "StorageBuffer" %cast_element : i32
              %float_old = spirv.Bitcast %old_bits : i32 to f32
              %float_next = spirv.FAdd %float_old, %float_one : f32
              %next_bits = spirv.Bitcast %float_next : f32 to i32
              spirv.Store "StorageBuffer" %cast_element, %next_bits : i32
              %j = spirv.IAdd %i, %one : i32
              spirv.Branch ^header(%j : i32)
            ^exit:
              spirv.mlir.merge
            }
            spirv.Return
          }
          spirv.EntryPoint "GLCompute" @main0
          spirv.ExecutionMode @main0 "LocalSize", 1, 1, 1
        }
      }
    }
  }
  hal.executable public @loop_calls {
    hal.executable.variant public @metal target(<"metal-spirv", "metal-msl-fb">) {
      hal.executable.export public @main0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
      builtin.module {
        spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
          spirv.GlobalVariable @input bind(0, 0) : !buffer
          spirv.GlobalVariable @read_write bind(0, 1) : !buffer
          spirv.func @update() "None" {
            %zero = spirv.Constant 0 : i32
            %input = spirv.mlir.addressof @input : !buffer
            %output = spirv.mlir.addressof @read_write : !buffer
            %in = spirv.AccessChain %input[%zero, %zero] : !buffer, i32, i32 -> !element
            %out = spirv.AccessChain %output[%zero, %zero] : !buffer, i32, i32 -> !element
            %value = spirv.Load "StorageBuffer" %in : i32
            %old = spirv.Load "StorageBuffer" %out : i32
            %sum = spirv.IAdd %old, %value : i32
            spirv.Store "StorageBuffer" %out, %sum : i32
            spirv.Return
          }
          spirv.func @main0() "None" {
            %zero = spirv.Constant 0 : i32
            %one = spirv.Constant 1 : i32
            %limit = spirv.Constant 11 : i32
            spirv.mlir.loop {
              spirv.Branch ^header(%zero : i32)
            ^header(%i: i32):
              %more = spirv.SLessThan %i, %limit : i32
              spirv.BranchConditional %more, ^body, ^exit
            ^body:
              spirv.FunctionCall @update() : () -> ()
              %next = spirv.IAdd %i, %one : i32
              spirv.Branch ^header(%next : i32)
            ^exit:
              spirv.mlir.merge
            }
            spirv.Return
          }
          spirv.EntryPoint "GLCompute" @main0
          spirv.ExecutionMode @main0 "LocalSize", 1, 1, 1
        }
      }
    }
  }
  hal.executable public @loop_pure_calls {
    hal.executable.variant public @metal target(<"metal-spirv", "metal-msl-fb">) {
      hal.executable.export public @main0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
      builtin.module {
        spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
          spirv.GlobalVariable @input bind(0, 0) : !buffer
          spirv.GlobalVariable @output bind(0, 1) : !buffer
          spirv.func @increment(%value: i32) -> i32 "None" {
            %one = spirv.Constant 1 : i32
            %next = spirv.IAdd %value, %one : i32
            spirv.ReturnValue %next : i32
          }
          spirv.func @main0() "None" {
            %zero = spirv.Constant 0 : i32
            %one = spirv.Constant 1 : i32
            %limit = spirv.Constant 11 : i32
            %input = spirv.mlir.addressof @input : !buffer
            %output = spirv.mlir.addressof @output : !buffer
            spirv.mlir.loop {
              spirv.Branch ^header(%zero : i32)
            ^header(%i: i32):
              %more = spirv.SLessThan %i, %limit : i32
              spirv.BranchConditional %more, ^body, ^exit
            ^body:
              %in = spirv.AccessChain %input[%zero, %i] : !buffer, i32, i32 -> !element
              %out = spirv.AccessChain %output[%zero, %i] : !buffer, i32, i32 -> !element
              %value = spirv.Load "StorageBuffer" %in : i32
              %next_value = spirv.FunctionCall @increment(%value) : (i32) -> i32
              spirv.Store "StorageBuffer" %out, %next_value : i32
              %next = spirv.IAdd %i, %one : i32
              spirv.Branch ^header(%next : i32)
            ^exit:
              spirv.mlir.merge
            }
            spirv.Return
          }
          spirv.EntryPoint "GLCompute" @main0
          spirv.ExecutionMode @main0 "LocalSize", 1, 1, 1
        }
      }
    }
  }
  hal.executable public @loop_pointer {
    hal.executable.variant public @metal target(<"metal-spirv", "metal-msl-fb">) {
      hal.executable.export public @main0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>)
      builtin.module {
        spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader, VariablePointersStorageBuffer], []> {
          spirv.GlobalVariable @read_write bind(0, 0) : !buffer
          spirv.func @main0() "None" {
            %zero = spirv.Constant 0 : i32
            %one = spirv.Constant 1 : i32
            %limit = spirv.Constant 11 : i32
            %buffer = spirv.mlir.addressof @read_write : !buffer
            %element = spirv.AccessChain %buffer[%zero, %zero] : !buffer, i32, i32 -> !element
            %other = spirv.AccessChain %buffer[%zero, %one] : !buffer, i32, i32 -> !element
            spirv.mlir.loop {
              spirv.Branch ^header(%zero : i32)
            ^header(%i: i32):
              %more = spirv.SLessThan %i, %limit : i32
              spirv.BranchConditional %more, ^body, ^exit
            ^body:
              %choose = spirv.SLessThan %i, %one : i32
              %pointer = spirv.Select %choose, %element, %other : i1, !element
              %old = spirv.Load "StorageBuffer" %pointer : i32
              %sum = spirv.IAdd %old, %one : i32
              spirv.Store "StorageBuffer" %pointer, %sum : i32
              %next = spirv.IAdd %i, %one : i32
              spirv.Branch ^header(%next : i32)
            ^exit:
              spirv.mlir.merge
            }
            spirv.Return
          }
          spirv.EntryPoint "GLCompute" @main0
          spirv.ExecutionMode @main0 "LocalSize", 1, 1, 1
        }
      }
    }
  }
  hal.executable public @loop_array_copy {
    hal.executable.variant public @metal target(<"metal-spirv", "metal-msl-fb">) {
      hal.executable.export public @main0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
      builtin.module {
        spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
          spirv.GlobalVariable @input bind(0, 0) : !array_buffer
          spirv.GlobalVariable @output bind(0, 1) : !array_buffer
          spirv.func @main0() "None" {
            %zero = spirv.Constant 0 : i32
            %one = spirv.Constant 1 : i32
            %limit = spirv.Constant 11 : i32
            %input = spirv.mlir.addressof @input : !array_buffer
            %output = spirv.mlir.addressof @output : !array_buffer
            spirv.mlir.loop {
              spirv.Branch ^header(%zero : i32)
            ^header(%i: i32):
              %more = spirv.SLessThan %i, %limit : i32
              spirv.BranchConditional %more, ^body, ^exit
            ^body:
              spirv.CopyMemory "StorageBuffer" %output, "StorageBuffer" %input : !spirv.struct<(!spirv.array<4xf32, stride=4> [0])>
              spirv.CopyMemory "StorageBuffer" %input, "StorageBuffer" %output : !spirv.struct<(!spirv.array<4xf32, stride=4> [0])>
              %next = spirv.IAdd %i, %one : i32
              spirv.Branch ^header(%next : i32)
            ^exit:
              spirv.mlir.merge
            }
            spirv.Return
          }
          spirv.EntryPoint "GLCompute" @main0
          spirv.ExecutionMode @main0 "LocalSize", 1, 1, 1
        }
      }
    }
  }
  hal.executable public @loop_vector {
    hal.executable.variant public @metal target(<"metal-spirv", "metal-msl-fb">) {
      hal.executable.export public @main0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>)
      builtin.module {
        spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
          spirv.GlobalVariable @read_write bind(0, 0) : !vector_buffer
          spirv.func @main0() "None" {
            %zero = spirv.Constant 0 : i32
            %one = spirv.Constant 1 : i32
            %limit = spirv.Constant 11 : i32
            %ones = spirv.Constant dense<1.0> : vector<4xf32>
            %buffer = spirv.mlir.addressof @read_write : !vector_buffer
            %element = spirv.AccessChain %buffer[%zero, %zero] : !vector_buffer, i32, i32 -> !spirv.ptr<vector<4xf32>, StorageBuffer>
            spirv.mlir.loop {
              spirv.Branch ^header(%zero : i32)
            ^header(%i: i32):
              %more = spirv.SLessThan %i, %limit : i32
              spirv.BranchConditional %more, ^body, ^exit
            ^body:
              %old = spirv.Load "StorageBuffer" %element : vector<4xf32>
              %sum = spirv.FAdd %old, %ones : vector<4xf32>
              spirv.Store "StorageBuffer" %element, %sum : vector<4xf32>
              %next = spirv.IAdd %i, %one : i32
              spirv.Branch ^header(%next : i32)
            ^exit:
              spirv.mlir.merge
            }
            spirv.Return
          }
          spirv.EntryPoint "GLCompute" @main0
          spirv.ExecutionMode @main0 "LocalSize", 1, 1, 1
        }
      }
    }
  }

}
