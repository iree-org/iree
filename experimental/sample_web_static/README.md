# Static Web Sample

This experimental sample demonstrates one way to target the web platform with
IREE. The output artifact is a web page containing an interactive MNIST digits
classifier.

## Quickstart

1. Install IREE's host tools (e.g. by building the `install` target with CMake)
2. Install the Emscripten SDK by
   [following these directions](https://emscripten.org/docs/getting_started/downloads.html)
3. Initialize your Emscripten environment (e.g. run `emsdk_env.bat`)
4. From this directory, run `bash ./build_static_emscripten_demo.sh`
    * You may need to set the path to your host tools install
5. Open the localhost address linked in the script output

To rebuild most parts of the demo (C runtime, sample HTML, CMake config, etc.),
just `control + C` to stop the local webserver and rerun the script.

## How it works

This [MNIST model](../../iree/samples/models/mnist.mlir), also used in the
[Vision sample](../../iree/samples/vision/), is compiled using the "static
library" output setting of IREE's compiler (see the
[Static library sample](../../iree/samples/static_library)). The resulting
`.h` and `.o` files are compiled together with `main.c`, while the `.vmfb` is
embedded into a C file that is similarly linked in.

[Emscripten](https://emscripten.org/) is used (via the `emcmake` CMake wrapper)
to compile the output binary into WebAssembly and JavaScript files.

The provided `index.html` file can be served together with the output `.js`
and `.wasm` files.

## Multithreading

TODO(scotttodd): this is incomplete - more changes are needed to the C runtime
