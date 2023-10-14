#!/usr/bin/env python
# coding=utf-8

def magic_compile(torch_model, data):
   import torch_mlir
   from iree import compiler as ireec
   from iree import runtime as ireert
   # module = torch_mlir.compile(torch_model, data,
   #       output_type=torch_mlir.OutputType.STABLEHLO, use_tracing=False)
   module = torch_mlir.compile(
        torch_model, data, output_type="linalg-on-tensors")
   INPUT_MLIR = str(module)
   compiled_flatbuffer = ireec.tools.compile_str(
       INPUT_MLIR,
       target_backends=["vulkan-spirv"])
   config = ireert.Config("vulkan")
   ctx = ireert.SystemContext(config=config)
   vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
   ctx.add_vm_module(vm_module)
   return ctx
def run(ctx, arg0):
   f = ctx.modules.module["forward"]
   result = f(arg0).to_host()
   return result
def Engine(torch_model, data):
   ctx = magic_compile(torch_model, data)
   return run(ctx, data)
