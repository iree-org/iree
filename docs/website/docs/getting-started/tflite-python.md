# TFLite via Python

The example below demonstrates downloading, compiling, and executing a TFLite
model using the Python API. This includes some initial setup to declare global
variables, download the sample module, and download the sample inputs.

Declaration of absolute paths for the sample repo and import all required libraries.
The default setup uses the CPU backend as the only target. This can be reconfigured
to select alternative targets.

```python
import iree.compiler.tflite as iree_tflite_compile
import iree.runtime as iree_rt
import numpy
import os
import urllib.request

from PIL import Image

workdir = "/tmp/workdir"
os.makedirs(workdir, exist_ok=True)

tfliteFile = "/".join([workdir, "model.tflite"])
jpgFile = "/".join([workdir, "input.jpg"])
tfliteIR = "/".join([workdir, "tflite.mlir"])
tosaIR = "/".join([workdir, "tosa.mlir"])
bytecodeModule = "/".join([workdir, "iree.vmfb"])

backends = ["dylib-llvm-aot"]
config = "dylib"
```

The TFLite sample model and input are downloaded locally.

```python
tfliteUrl = "https://storage.googleapis.com/iree-model-artifacts/tflite-integration-tests/posenet_i8.tflite"
jpgUrl = "https://storage.googleapis.com/iree-model-artifacts/tflite-integration-tests/posenet_i8_input.jpg"

urllib.request.urlretrieve(tfliteUrl, tfliteFile)
urllib.request.urlretrieve(jpgUrl, jpgFile)
```

Once downloaded we can compile the model for the selected backends. Both the TFLite and TOSA representations
of the model are saved for debugging purposes. This is optional and can be omitted.

```python
iree_tflite_compile.compile_file(
  tfliteFile,
  input_type="tosa",
  output_file=bytecodeModule,
  save_temp_tfl_input=tfliteIR,
  save_temp_iree_input=tosaIR,
  target_backends=backends,
  import_only=False)
```

After compilation is completed we configure the VmModule using the dylib configuration and compiled
IREE module.

```python
config = iree_rt.Config("dylib")
context = iree_rt.SystemContext(config=config)
with open(bytecodeModule, 'rb') as f:
  vm_module = iree_rt.VmModule.from_flatbuffer(f.read())
  context.add_vm_module(vm_module)
```

Finally, the IREE module is loaded and ready for execution. Here we load the sample image, manipulate to
the expected input size, and execute the module. By default TFLite models include a single
function named 'main'. The final results are printed.

```python
im = numpy.array(Image.open(jpgFile).resize((192, 192))).reshape((1, 192, 192, 3))
args = [im]

invoke = context.modules.module["main"]
iree_results = invoke(*args)
print(iree_results)
```
