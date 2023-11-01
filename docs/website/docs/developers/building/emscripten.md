# Building with Emscripten

[Emscripten](https://emscripten.org/index.html) is a complete compiler
toolchain to WebAssembly, using LLVM, with a special focus on speed, size, and
the Web platform. Emscripten can be used to compile parts of IREE to
[WebAssembly](https://webassembly.org/) for execution within web browsers or
other Wasm runtimes.

## Status

IREE's _runtime_ can be compiled through Emscripten in some limited
configurations. More of the runtime will be supported over time.

IREE's _compiler_ can be compiled through Emscripten with local changes. More
work is needed for this to be generally supported.

## Prerequisites

Read <https://emscripten.org/docs/getting_started/downloads.html> and run

```shell
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

## Building IREE's runtime with Emscripten

### Host configuration

Build and install at least the compiler tools on your host machine, or install
them from a binary distribution:

```shell
$ cmake -G Ninja -B ../iree-build-host/ \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX=../iree-build-host/install \
    .
$ cmake --build ../iree-build-host/ --target install
```

### Target configuration

```shell
$ emcmake cmake -G Ninja -B ../iree-build-emscripten/ \
  -DCMake_BUILD_TYPE=Release \
  -DIREE_HOST_BIN_DIR=$(realpath ../iree-build-host/install/bin) \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_COMPILER=OFF \
  .
```

Build:

```shell
cmake --build ../iree-build-emscripten/ \
  --target iree_samples_simple_embedding_simple_embedding_vmvx_sync
```

### Load into a WebAssembly environment

Copy the outputs from the build process (e.g. `simple_embedding_vmvx_sync.js`
and `simple_embedding_vmvx_sync.wasm`) into your application and follow
instructions at either <https://webassembly.org/getting-started/developers-guide/>
or <https://developer.mozilla.org/en-US/docs/WebAssembly/Loading_and_running>.
