# Static Web Sample

This experimental sample demonstrates one way to target the web platform with
IREE. The output artifact is a web page containing an interactive MNIST digits
classifier.

The MNIST ML model is compiled statically together with the IREE runtime into
a single .js + .wasm bundle.

## Quickstart

1. Install IREE's host tools (e.g. by building the `install` target with CMake)
2. Install the Emscripten SDK by
   [following these directions](https://emscripten.org/docs/getting_started/downloads.html)
3. Initialize your Emscripten environment (e.g. run `emsdk_env.bat`)
4. From this directory, run `bash ./build_sample.sh [path to install] && bash ./serve_sample.sh`
5. Open the localhost address linked in the script output

To rebuild most parts of the sample (C runtime, sample HTML, CMake config,
etc.), just `control + C` to stop the local webserver and rerun the script.

## How it works

This [MNIST model](../../../samples/models/mnist.mlir), also used in the
[Vision sample](../../../samples/vision_inference/), is compiled using the "static
library" output setting of IREE's compiler (see the
[Static library sample](../../../samples/static_library)). The resulting
`.h` and `.o` files are compiled together with `main.c`, while the `.vmfb` is
embedded into a C file that is similarly linked in.

[Emscripten](https://emscripten.org/) is used (via the `emcmake` CMake wrapper)
to compile the output binary into WebAssembly and JavaScript files.

The provided [`index.html`](./index.html) file can be served together with the
output `.js` and `.wasm` files.

### Asynchronous API

* [`iree_api.js`](./iree_api.js) exposes a Promise-based API to the hosting
  application in [`index.html`](./index.html)
* [`iree_api.js`](./iree_api.js) creates a worker running iree_worker.js, which
  includes Emscripten's JS code and instantiates the WebAssembly module
* messages are passed back and forth between [`iree_api.js`](./iree_api.js) and
  [`iree_worker.js`](./iree_worker.js) internally

### Multithreading

The sample supports running both single-threaded using
[`device_sync.c`](./device_sync.c) (backed by the local HAL's 'sync' device)
and multi-threaded using [`device_multithreaded.c`](./device_multithreaded.c)
(backed by the local HAL's 'task' device).

Each configuration is offered as a CMake target, then
[`iree_worker.js`](./iree_worker.js) specifies which script URL to load.

Multithreading requires Web Workers and SharedArrayBuffer:

* https://caniuse.com/webworkers
* https://caniuse.com/sharedarraybuffer
