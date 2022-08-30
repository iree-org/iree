# WebGPU Sample

This experimental sample demonstrates one way to target the web platform with
IREE, using WebGPU. The output artifact is a web page that loads separately
provided IREE `.vmfb` (compiled ML model) files and allows for calling
functions on them.

## Quickstart

**Note**: you will need a WebGPU-compatible browser. Chrome Canary with the
`#enable-unsafe-webgpu` flag is a good choice (you may need the flag or an
origin trial token for `localhost`).

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
the WebGPU compiler target. This generates WGSL shader code and IREE VM
bytecode, which the IREE runtime is able to load and run using the browser's
WebGPU APIs.

### Asynchronous API

[`iree_api_webgpu.js`](./iree_api_webgpu.js)

* exposes a Promise-based API to the hosting application in
  [`index.html`](./index.html)
* preinitializes a WebGPU adapter and device
* includes Emscripten's JS code and instantiates the WebAssembly module
