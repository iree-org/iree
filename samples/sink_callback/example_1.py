#!/usr/bin/env python3
import iree.runtime as rt
import iree.compiler as compiler
import numpy as np

# softmax(matmul(arg0, arg1))
MLIR_MODEL = r"""
func.func @main(%arg0: tensor<?x16xf32>, %arg1: tensor<16x16xf32>)
    -> tensor<?x16xf32> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %arg0, %c0 : tensor<?x16xf32>
  %empty = tensor.empty(%m) : tensor<?x16xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x16xf32>)
               -> tensor<?x16xf32>
  %mat = linalg.matmul ins(%arg0, %arg1 : tensor<?x16xf32>, tensor<16x16xf32>)
                      outs(%init : tensor<?x16xf32>) -> tensor<?x16xf32>
  %out = linalg.softmax dimension(1) ins(%mat : tensor<?x16xf32>)
                        outs(%empty : tensor<?x16xf32>) -> tensor<?x16xf32>
  return %out : tensor<?x16xf32>
}
"""


def callback(key: str, buffer_views: list[rt.HalBufferView]):
    print(f"--- Debug callback for key={key}  ---")
    for i, bv in enumerate(buffer_views):
        arr = bv.map().asarray(
            bv.shape, rt.HalElementType.map_to_dtype(bv.element_type)
        )
        print(f"  Tensor {i}: shape={arr.shape}, mean={arr.mean():.6f}")


def main():
    # TODO(newling) better way to specify target generically.
    compiler.compile_str(
        MLIR_MODEL,
        output_file="simple.vmfb",
        extra_args=[
            "--iree-hal-target-device=hip[0]",
            "--iree-hip-target=gfx942",
            "--iree-flow-trace-dispatch-tensors",  # <-- crucial to emit traces
        ],
    )

    # Set up configuration + debug sink
    config = rt.Config("hip")
    hal_module = rt.create_hal_module(
        config.vm_instance,
        config.device,
        debug_sink=rt.HalModuleDebugSink(callback),
    )

    # Load modules
    with open("simple.vmfb", "rb") as f:
        module = rt.VmModule.from_buffer(config.vm_instance, f.read())
    vm_modules = rt.load_vm_modules(hal_module, module, config=config)

    # Run function
    A = np.random.rand(4, 16).astype(np.float32)
    B = np.random.rand(16, 16).astype(np.float32)
    print("Running main function.")
    vm_modules[-1].main(A, B)
    print("Done.")


if __name__ == "__main__":
    main()
