#!/usr/bin/env python
# coding=utf-8

def magic_compile(torch_model, *args):

    import torch_mlir
    from iree import compiler as ireec
    from iree import runtime as ireert

    # compile torchmodel to mlir
    module = torch_mlir.compile(
         torch_model, *args, output_type="linalg-on-tensors", use_tracing=False)
    INPUT_MLIR = str(module)

    # compile mlir to flabbuffer
    compiled_flatbuffer = ireec.tools.compile_str(
        INPUT_MLIR,
        target_backends=["vulkan-spirv"])

    # runtime load flatbuffer
    config = ireert.Config("vulkan")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, compiled_flatbuffer)
    ctx.add_vm_module(vm_module)
    return ctx


def run(ctx, *args):

    f = ctx.modules.module["forward"]
    result = f(*args).to_host()
    return result


def Engine(torch_model, *args):

    ctx = magic_compile(torch_model, args)
    res = run(ctx, *args)
    return res
