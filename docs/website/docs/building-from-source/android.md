---
hide:
  - tags
tags:
  - Android
---

# Android cross-compilation

Running on a platform like Android involves cross-compiling from a _host_
platform (e.g. Linux) to a _target_ platform (a specific Android version and
system architecture):

* IREE's _compiler_ is built on the host and is used there to generate modules
  for the target
* IREE's _runtime_ is built on the host for the target. The runtime is then
  either pushed to the target to run natively or is bundled into an Android
  [APK](https://en.wikipedia.org/wiki/Android_application_package)

## Prerequisites

### Host environment setup

You should already be able to build IREE from source on your host platform.
Please make sure you have followed the [getting started](./getting-started.md)
steps.

### Install Android NDK and ADB

The Android [Native Developer Kit (NDK)](https://developer.android.com/ndk) is
needed to use native C/C++ code on Android. You can
[download it here](https://developer.android.com/ndk/downloads), or, if you
have installed [Android Studio](https://developer.android.com/studio), you can
follow [this guide](https://developer.android.com/studio/projects/install-ndk)
instead.

!!! note
    Make sure the `ANDROID_NDK` environment variable is set after installing
    the NDK.

ADB (the Android Debug Bridge) is also needed to communicate with Android
devices from the command line. Install it following the
[official user guide](https://developer.android.com/studio/command-line/adb).

## Configure and build

### Host configuration

Build and install on your host machine:

``` shell
cmake -GNinja -B ../iree-build/ \
  -DCMAKE_INSTALL_PREFIX=../iree-build/install \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  .
cmake --build ../iree-build/ --target install
```

### Target configuration

Build the runtime using the Android NDK toolchain:

=== "Linux"

    ``` shell
    cmake -GNinja -B ../iree-build-android/ \
      -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK?}/build/cmake/android.toolchain.cmake" \
      -DIREE_HOST_BIN_DIR="$PWD/../iree-build/install/bin" \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_PLATFORM="android-29" \
      -DIREE_BUILD_COMPILER=OFF \
      .
    cmake --build ../iree-build-android/
    ```

=== "macOS"

    ``` shell
    cmake -GNinja -B ../iree-build-android/ \
      -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK?}/build/cmake/android.toolchain.cmake" \
      -DIREE_HOST_BIN_DIR="$PWD/../iree-build/install/bin" \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_PLATFORM="android-29" \
      -DIREE_BUILD_COMPILER=OFF \
      .
    cmake --build ../iree-build-android/
    ```

=== "Windows"

    ``` shell
    cmake -GNinja -B ../iree-build-android/ \
      -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK%/build/cmake/android.toolchain.cmake" \
      -DIREE_HOST_BIN_DIR="%CD%/../iree-build/install/bin" \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_PLATFORM="android-29" \
      -DIREE_BUILD_COMPILER=OFF \
      .
    cmake --build ../iree-build-android/
    ```

!!! note
    See the
    [Android NDK CMake guide](https://developer.android.com/ndk/guides/cmake)
    and
    [Android Studio CMake guide](https://developer.android.com/studio/projects/configure-cmake)
    for details on configuring CMake for Android.

    The specific `ANDROID_ABI` and `ANDROID_PLATFORM` used should match your
    target device.

## Running Android tests

Make sure you
[enable developer options and USB debugging](https://developer.android.com/studio/debug/dev-options#enable)
on your Android device and can see your it when you run `adb devices`, then run
all built tests through
[CTest](https://gitlab.kitware.com/cmake/community/-/wikis/doc/ctest/Testing-With-CTest):

``` shell
ctest --test-dir ../iree-build-android/ --output-on-failure
```

This will automatically upload build artifacts to the connected Android device,
run the tests, then report the status back to your host machine.

## Running tools directly

Invoke the host compiler tools to produce a bytecode module FlatBuffer:

``` shell
../iree-build/install/bin/iree-compile \
  --iree-hal-target-backends=vmvx \
  samples/models/simple_abs.mlir \
  -o /tmp/simple_abs_vmvx.vmfb
```

Push the Android runtime tools to the device, along with any FlatBuffer files:

``` shell
adb push ../iree-build-android/tools/iree-run-module /data/local/tmp/
adb shell chmod +x /data/local/tmp/iree-run-module
adb push /tmp/simple_abs_vmvx.vmfb /data/local/tmp/
```

Run the tool:

``` shell
adb shell /data/local/tmp/iree-run-module --device=local-task \
  --module=/data/local/tmp/simple_abs_vmvx.vmfb \
  --function=abs \
  --input="f32=-5"
```
