# Dynamic Web Sample

This experimental sample demonstrates one way to target the web platform with
IREE. The output artifact is a web page that loads separately provided IREE
`.vmfb` (compiled ML model) files and allows for calling functions on them.

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

[Emscripten](https://emscripten.org/) is used (via the `emcmake` CMake wrapper)
to compile the runtime into WebAssembly and JavaScript files.

Any supported IREE program, such as
[simple_abs.mlir](../../../samples/models/simple_abs.mlir), is compiled using
the "system library" linking mode. This creates a shared object (typically
.so/.dll, .wasm in this case). When the runtime attempts to load this file
using `dlopen()` and `dlsym()`, Emscripten makes use of its
[runtime dynamic linking support](https://emscripten.org/docs/compiling/Dynamic-Linking.html#runtime-dynamic-linking-with-dlopen)
to instantiate a new `WebAssembly.Instance` which shares memory with the main
runtime then resolve each export provided by the new Wasm module.

### Asynchronous API

* [`iree_api.js`](./iree_api.js) exposes a Promise-based API to the hosting
  application in [`index.html`](./index.html)
* [`iree_api.js`](./iree_api.js) creates a worker running iree_worker.js, which
  includes Emscripten's JS code and instantiates the WebAssembly module
* messages are passed back and forth between [`iree_api.js`](./iree_api.js) and
  [`iree_worker.js`](./iree_worker.js) internally

### Multithreading

Multithreading is _not supported yet_. Emscripten only has experimental support
for dynamic linking + pthreads:
https://emscripten.org/docs/compiling/Dynamic-Linking.html#pthreads-support.
Compiled programs produced by IREE link with `wasm-ld`, while Emscripten expects
programs to be linked using `emcc` with the `-s SIDE_MODULE` option, which
includes several Emscripten-pthreads-specific module exported functions such as
`emscripten_tls_init`.
