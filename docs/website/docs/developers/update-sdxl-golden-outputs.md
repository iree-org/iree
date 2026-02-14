---
icon: material/lightbulb-on
---

# Updating SDXL Golden Outputs for IREE CI

Golden outputs are reference results generated from a known-good version of the
SDXL pipeline. They serve as the “ground truth” for CI quality tests in IREE,
ensuring that future changes do not silently alter accuracy. When a change is
made which affects the numerics (e.g, modifying the order of floating-point
operations), differences in outputs can occur. In such cases, you must
regenerate the golden outputs so that CI reflects the new expected results. This
page describes the end-to-end process: verifying accuracy, generating new
outputs, uploading them to storage, bumping the version in configuration, and
re-running CI.

## Verify accuracy before updating goldens

Before updating golden outputs, first confirm your change maintains acceptable
accuracy. Follow the steps
[outlined](https://github.com/nod-ai/AMD-SHARK-MLPERF/blob/dev/code/stable-diffusion-xl/development.md#test-accuracy-only).
Use the offline variant of the `precompile_model_shortfin.sh` script for your
platform. On MI300X use the one for MI325X.

A straightforward way to test your change is by editing
`sdxl_harness_rocm_shortfin_from_source_iree.dockerfile` so that it builds your
IREE and exposes the right tooling:

- Build your IREE commit and add the build’s tools to `PATH`.
- Add your IREE Python bindings to `PYTHONPATH`.
- Remove the prebuilt wheels for `iree-base-compiler` and `iree-base-runtime` so
  you’re testing your own build.

Run the accuracy script (`run_accuracy_mi325x.sh`) and be mindful of
platform-specific settings. If you are running in SPX mode, update available
device IDs accordingly (i.e., change to `DEVICES="0,1,2,3,4,5,6,7"`). On MI300x,
set `CPD=1` and use `BATCH_SIZE=32`. Accuracy is considered acceptable if FID
and CLIP scores fall within the advertised ranges.

## Generate new outputs with your IREE build

Once accuracy is confirmed, generate new outputs using the same inputs that CI
consumes. Both inputs and outputs live in the `sharkpublic` Azure container. If
you do not already have the desired inputs, locate and download the input files
for your model revision and place them in a local directory. You may find the
exact paths in the relevant json file in
`tests/external/iree-test-suites/sharktank_models/quality_tests/sdxl/`.

Next, compile the relevant model using your IREE build. The exact flags should
mirror what CI uses for the target you're validating. You can find this
information from failing CI logs or from the same json file as mentioned above.
The example below shows a representative invocation; replace paths and flags
with your local equivalents as needed.

```bash
iree-build/tools/iree-compile \
  -o model.rocm_gfx942.vmfb \
  punet_fp16.mlir \
  --mlir-timing \
  --mlir-timing-display=list \
  --iree-consteval-jit-debug \
  --iree-hal-target-device=hip \
  --iree-opt-const-eval=false \
  --iree-opt-level=O3 \
  --iree-dispatch-creation-enable-fuse-horizontal-contractions=true \
  --iree-vm-target-truncate-unsupported-floats \
  --iree-llvmgpu-enable-prefetch=true \
  --iree-opt-data-tiling=false \
  --iree-codegen-gpu-native-math-precision=true \
  --iree-codegen-llvmgpu-use-vector-distribution \
  --iree-rocm-waves-per-eu=2 \
  --iree-execution-model=async-external \
  --iree-scheduling-dump-statistics-format=json \
  --iree-scheduling-dump-statistics-file=compilation_info.json \
  --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-flow-canonicalize), iree-preprocessing-transpose-convolution-pipeline, iree-preprocessing-pad-to-intrinsics)" \
  --iree-codegen-transform-dialect-library=/path/to/attention_and_matmul_spec_punet_mi300.mlir \
  --iree-rocm-target=gfx942
```

After compilation, run the module to produce the new outputs that will become
the new goldens:

```bash
iree-build/tools/iree-run-module \
  --device=hip \
  --module=model.rocm_gfx942.vmfb \
  --function=main \
  --input=1x4x128x128xf16=@${CACHE_DIR}/punet_input0.bin \
  --input=1xf16=@${CACHE_DIR}/punet_input1.bin \
  --input=2x64x2048xf16=@${CACHE_DIR}/punet_input2.bin \
  --input=2x1280xf16=@${CACHE_DIR}/punet_input3.bin \
  --input=2x6xf16=@${CACHE_DIR}/punet_input4.bin \
  --input=1xf16=@${CACHE_DIR}/punet_input5.bin \
  --parameters=model=/path/to/punet_weights.irpa \
  --output=@punet_fp16_out_v{n+1}.0.bin
```

## Upload new outputs to Azure

With outputs generated, upload the new `v{n+1}` outputs to the same location in
the `sharkpublic` Azure container as the previous outputs.

```bash
az storage blob upload \
  --account-name sharkpublic \
  --container-name sharkpublic \
  --name <path/in/blob/container> \
  --file <local/file/path>
```

After uploading, update the configuration that tells CI which golden version to
use. This is typically a JSON key whose value encodes the version (for example,
`punet_output_v{n}`). Increment it to `punet_output_v{n+1}` and commit this
change along with any related edits.

Finally, re-run the CI pipeline and confirm the quality tests pass against the
newly uploaded outputs.
