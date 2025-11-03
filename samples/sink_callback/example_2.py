#!/usr/bin/env python3
import argparse
import time
import iree.runtime as rt
import iree.compiler as compiler
import numpy as np
from pathlib import Path
import logging
import threading
from typing import Callable

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def get_summary_statistics(arr: np.ndarray):
    """
    Computes minimal summary statistics for a tensor mapped back to host.
    NB: keep this lightweight — it runs on CPU and can easily bottleneck the
    debug pipeline.
    """
    return {"mean": arr.mean()}


class MeanLogger:
    """
    Log the mean of np.arrays to a file. It is very likely that the dispatch
    that introduces a numerical change will have a difference spike here.
    """

    def __init__(self, model_path: Path, f_log):
        self.gathered_stats: list[dict] = []
        self.callback_time_total = 0.0
        self.start_total = time.perf_counter()
        self.last_total_print = 0.0
        self.model_path = model_path
        self.f_log = f_log

    def log_and_dump(self, message: str):
        log.info(f"{self.model_path.name} | {message}")
        self.f_log.write(message + "\n")

    def callback(self, key: str, buffer_views: list[rt.HalBufferView]):
        start = time.perf_counter()

        for i, bv in enumerate(buffer_views):
            arr = bv.map().asarray(
                bv.shape, rt.HalElementType.map_to_dtype(bv.element_type)
            )
            ss = get_summary_statistics(arr)
            ss.update({"count": len(self.gathered_stats), "key": key, "index": i})
            self.gathered_stats.append(ss)
            self.log_and_dump(
                f"BufferView {ss['count']:04d} | Key: {key} | "
                f"Index: {i} | Mean: {ss['mean']:.8f}"
            )

        self.callback_time_total += time.perf_counter() - start
        total_time = time.perf_counter() - self.start_total
        if total_time - self.last_total_print > 0.5:
            log.info(
                f"{self.model_path.name} | Total: {total_time:.3f}s | "
                f"Callback: {self.callback_time_total:.3f}s "
                f"({self.callback_time_total / total_time * 100:.1f}%)"
            )
            self.last_total_print = total_time


def run(
    model_vmfb_filename: Path,
    driver_vmfb_filename: Path,
    callback: Callable[[str, list[rt.HalBufferView]], None],
    config: rt.Config,
    executor: Callable[[rt.VmModule], None],
    weight_provider: rt.ParameterProvider,
):
    """
    Runs a model with a HAL debug callback that logs simple statistics
    for every BufferView that is an input/output to a dispatch.
    """
    model_path = Path(model_vmfb_filename)
    driver_path = Path(driver_vmfb_filename)

    with (
        open(model_path, "rb") as f_model,
        open(driver_path, "rb") as f_driver,
    ):
        model_module = rt.VmModule.from_buffer(config.vm_instance, f_model.read())
        driver_module = rt.VmModule.from_buffer(config.vm_instance, f_driver.read())
        io_module = rt.create_io_parameters_module(config.vm_instance, weight_provider)

        hal_module = rt.create_hal_module(
            config.vm_instance,
            config.device,
            debug_sink=rt.HalModuleDebugSink(callback),
        )

        vm_modules = rt.load_vm_modules(
            hal_module, io_module, model_module, driver_module, config=config
        )

        log.info(f"Running {model_path.name}")
        executor(vm_modules[-1])


def example_2():
    """Compiles two similar MLIR models, and logs statistics (means)."""
    config = rt.Config("hip")

    model_a_mlir = r"""
#map = affine_map<(d0, d1) -> (d0, d1)>
module @foo {
  util.global private @a0 = #flow.parameter.named<"my_scope"::"my_weights"> : tensor<18x18xf32>
  func.func @foo(%arg0: tensor<?x18xf32>, %arg1: tensor<18x18xf32>) -> tensor<?x18xf32> {
    %a0 = util.global.load @a0 : tensor<18x18xf32>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x18xf32>
    %0 = tensor.empty(%dim) : tensor<?x18xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x18xf32>) -> tensor<?x18xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<?x18xf32>, tensor<18x18xf32>) outs(%1 : tensor<?x18xf32>) -> tensor<?x18xf32>
    %3 = linalg.matmul ins(%2, %a0 : tensor<?x18xf32>, tensor<18x18xf32>) outs(%1 : tensor<?x18xf32>) -> tensor<?x18xf32>
    %4 = linalg.softmax dimension(1) ins(%3 : tensor<?x18xf32>) outs(%1 : tensor<?x18xf32>) -> tensor<?x18xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<?x18xf32>) outs(%0 : tensor<?x18xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_0 = arith.constant 3.1415e+00 : f32
      %6 = arith.addf %in, %cst_0 : f32
      linalg.yield %6 : f32
    } -> tensor<?x18xf32>
    return %5 : tensor<?x18xf32>
  }
}"""

    ### Model B is very similar to model A, except for a constant change
    model_b_mlir = model_a_mlir.replace("3.1415", "2.7182")

    driver_mlir = r"""
module @driver {
  util.func private @foo.foo(%arg0: tensor<?x18xf32>, %arg1: tensor<18x18xf32>) -> tensor<?x18xf32>
  util.func public @main(%arg0: tensor<?x18xf32>, %arg1: tensor<18x18xf32>) {
    %0 = util.call @foo.foo(%arg0, %arg1) : (tensor<?x18xf32>, tensor<18x18xf32>) -> tensor<?x18xf32>
    util.return
  }
}"""

    artifacts_dir = Path(__file__).parent / "example_2_artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    irpa_path = artifacts_dir / "archive.irpa"
    a_vmfb_path = artifacts_dir / "model_a.vmfb"
    b_vmfb_path = artifacts_dir / "model_b.vmfb"
    driver_vmfb_path = artifacts_dir / "driver.vmfb"
    output_a_log = artifacts_dir / "debug_output_a.txt"
    output_b_log = artifacts_dir / "debug_output_b.txt"

    for mlir_str, vmfb_fn in [
        (model_a_mlir, a_vmfb_path),
        (model_b_mlir, b_vmfb_path),
        (driver_mlir, driver_vmfb_path),
    ]:
        compiler.compile_str(
            mlir_str,
            output_file=str(vmfb_fn),
            target_backends=["rocm"],
            extra_args=[
                "--iree-hal-target-device=hip[0]",
                "--iree-hip-target=gfx942",
                "--iree-flow-trace-dispatch-tensors",
            ],
        )

    vals = np.arange(18 * 18, dtype=np.float32).reshape((18, 18)) / (18 * 18)
    rt.save_archive_file({"my_weights": vals}, irpa_path)
    params = rt.ParameterIndex()
    params.load(str(irpa_path), format="irpa")
    provider = params.create_provider(scope="my_scope")

    def test_executor(module: rt.VmModule):
        A = np.ones((4, 18), dtype=np.float32)
        B = np.ones((18, 18), dtype=np.float32)
        module.main(A, B)

    with (
        open(output_a_log, "w", buffering=1) as a_file,
        open(output_b_log, "w", buffering=1) as b_file,
    ):
        mean_logger_a = MeanLogger(a_vmfb_path, a_file)
        mean_logger_b = MeanLogger(b_vmfb_path, b_file)
        threads = []
        for model_path, callback in [
            (a_vmfb_path, mean_logger_a.callback),
            (b_vmfb_path, mean_logger_b.callback),
        ]:
            t = threading.Thread(
                target=run,
                args=(
                    model_path,
                    driver_vmfb_path,
                    callback,
                    config,
                    test_executor,
                    provider,
                ),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    print("Both model runs complete.")


if __name__ == "__main__":
    example_2()
