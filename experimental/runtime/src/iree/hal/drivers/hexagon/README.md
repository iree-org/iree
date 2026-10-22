# Qualcomm Hexagon DSP HAL driver (`hexagon`)

**Experimental.** A HAL driver running programs on the Hexagon DSP of Qualcomm
Snapdragon SoCs from an Android aarch64 host via FastRPC.

Status: the driver lists the DSP domains of the device and creates devices,
which opens a FastRPC session with the DSP. Buffers, executables, command
buffers, and queue operations are not implemented yet and fail with
`UNIMPLEMENTED`.

## Directory structure

- `.`: host side of the driver (Android aarch64).
- `registration/`: registration of the driver with the HAL driver registry.
- `interface/`: FastRPC interface definition (`hexagon_dsp.idl`). The Hexagon
  SDK `qaic` compiler generates the host stub and the DSP skeleton from it.
- `dsp/`: DSP side, built as `libhexagon_dsp_skel.so` for Hexagon v79.

## Prerequisites

- An Android device with a Snapdragon SoC whose compute DSP (CDSP) allows
  unsigned protection domains (e.g. Snapdragon 8 Elite / SM8750).
- [Android NDK r28](https://developer.android.com/ndk/downloads) (tested with
  r28c, `28.2.13676358`) with `ANDROID_NDK` set to its location.
- The [Qualcomm Hexagon SDK](https://github.com/snapdragon-toolchain/hexagon-sdk/releases)
  (free download, see its license). Tested with 6.7.0.0, which Bazel downloads
  automatically, and 6.4.0.2.

## Building

### Bazel

```shell
export ANDROID_NDK=/path/to/android-ndk-r28c
bazel build --config=hexagon \
  //tools:iree-run-module \
  //experimental/runtime/src/iree/hal/drivers/hexagon/dsp:hexagon_dsp_skel
```

`--config=hexagon` builds for Android aarch64 and links the driver into the
tools. To use an existing Hexagon SDK installation instead of downloading it,
add `--repo_env=HEXAGON_SDK_ROOT=/path/to/hexagon-sdk`.

### CMake

Build the host tools as described in the
[Android cross-compilation guide](https://iree.dev/building-from-source/android/),
then configure the Android build with the driver enabled:

```shell
cmake -GNinja -B ../iree-build-android/ \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK?}/build/cmake/android.toolchain.cmake" \
  -DIREE_HOST_BIN_DIR="$PWD/../iree-build/install/bin" \
  -DANDROID_ABI="arm64-v8a" \
  -DANDROID_PLATFORM="android-28" \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_EXTERNAL_HAL_DRIVERS=hexagon \
  -DHEXAGON_SDK_ROOT=/path/to/hexagon-sdk \
  .
cmake --build ../iree-build-android/ --target \
  iree-run-module iree_hal_drivers_hexagon_dsp_hexagon_dsp_skel
```

The DSP library is built at
`experimental/runtime/src/iree/hal/drivers/hexagon/dsp/libhexagon_dsp_skel.so`
in the build directory.

## Deployment

The driver loads the DSP library from `../lib/hexagon/` relative to the
directory of the running executable:

```text
<dir>/bin/iree-run-module
<dir>/lib/hexagon/libhexagon_dsp_skel.so
<dir>/lib/libc++_shared.so   (Bazel builds only, from the NDK sysroot)
```

```shell
adb shell mkdir -p /data/local/tmp/iree/bin /data/local/tmp/iree/lib/hexagon
adb push iree-run-module /data/local/tmp/iree/bin/
adb push libhexagon_dsp_skel.so /data/local/tmp/iree/lib/hexagon/
adb push "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" \
  /data/local/tmp/iree/lib/
```

## Devices

Each DSP domain that supports unsigned protection domains is a device:

```text
$ iree-run-module --list_devices=hexagon
hexagon://CDSP
```

Devices are selected by domain name (`hexagon://CDSP`) or domain ID
(`hexagon://3`). `hexagon://` selects the CDSP. Device parameters:

| Parameter | Description |
| --- | --- |
| `verbose=1` | Prints diagnostics such as the FastRPC session URI to stderr. |
| `dsp_status_notify=1` | Prints DSP protection domain status changes to stderr. |

Example: `--device="hexagon://CDSP?verbose=1&dsp_status_notify=1"`.

## Manual verification

There are no automated tests yet. On a device:

1. Listing devices:

    ```shell
    adb shell 'cd /data/local/tmp/iree && bin/iree-run-module --list_devices=hexagon'
    ```

    Expected output: `hexagon://CDSP`.

2. Opening a device: `--list_devices` does not create devices. The tools only
   create devices for modules that import the `hal` module, so compile a small
   module for any target on the host:

    ```mlir
    func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
      return %0 : tensor<4xf32>
    }
    ```

    ```shell
    iree-compile --iree-hal-target-device=local \
      --iree-hal-local-target-device-backends=vmvx add.mlir -o add.vmfb
    adb push add.vmfb /data/local/tmp/iree/
    adb shell 'cd /data/local/tmp/iree && bin/iree-run-module \
      --device="hexagon://CDSP?verbose=1&dsp_status_notify=1" \
      --module=add.vmfb --function=main --input=4xf32=1,2,3,4'
    ```

    Expected output: the device opens the session, the DSP library is loaded
    (`adb logcat` shows `libhexagon_dsp_skel.so` being opened), and the run
    then fails as executables are not implemented yet:

    ```text
    hexagon: opening DSP session 'file:////data/local/tmp/iree/bin/../lib/hexagon/libhexagon_dsp_skel.so?hexagon_dsp_skel_handle_invoke&_modver=1.0&_idlver=0.1.0&_dom=cdsp'
    hexagon: DSP CDSP status: PD is up
    hexagon: DSP CDSP status: PD closed
    hexagon: closed DSP session (result 0x0)
    .../hexagon/device.c:...: UNIMPLEMENTED; hexagon executables not implemented; creating VM context; creating run context
    ```
