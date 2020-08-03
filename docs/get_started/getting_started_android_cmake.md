# Getting Started on Android with CMake

<!--
Notes to those updating this guide:

    * This document should be __simple__ and cover essential items only.
      Notes for optional components should go in separate files.
-->

This guide walks through cross-compiling IREE core runtime towards the Android
platform. Cross-compiling IREE compilers towards Android is not supported at the
moment.

Cross-compilation involves both a *host* platform and a *target* platform. One
invokes compiler toolchains on the host platform to generate libraries and
executables that can be run on the target platform.

## Prerequisites

### Set up host development environment

The host platform should have been set up for developing IREE. Right now Linux
and Windows are supported. Please make sure you have followed the steps for
[Linux](./getting_started_linux_cmake.md) or
[Windows](./getting_started_windows_cmake.md).

### Install Android NDK

Android NDK provides compiler toolchains for compiling C/C++ code to target
Android. You can download it
[here](https://developer.android.com/ndk/downloads). We recommend to download
the latest release; the steps in following sections may assume that.

Alternatively, if you have installed
[Android Studio](https://developer.android.com/studio), you can follow
[this guide](https://developer.android.com/studio/projects/install-ndk) to
install Android NDK.

After downloading, it is recommended to set the `ANDROID_NDK` environment
variable pointing to the directory. For Linux, you can `export` in your shell's
rc file. For Windows, you can search "environment variable" in the taskbar or
use `Windows` + `R` to open the "Run" dialog to run `rundll32
sysdm.cpl,EditEnvironmentVariables`.

### Install Android Debug Bridge (ADB)

For Linux, search your the distro's package manager to install `adb`. For
example, on Ubuntu:

```shell
$ sudo apt install adb
```

For Windows, it's easier to get `adb` via Android Studio. `adb` is included in
the Android SDK Platform-Tools package. You can download this package with the
[SDK Manager](https://developer.android.com/studio/intro/update#sdk-manager),
which installs it at `android_sdk/platform-tools/`. Or if you want the
standalone Android SDK Platform-Tools package, you can
[download it here](https://developer.android.com/studio/releases/platform-tools).
You may also want to add the folder to the `PATH` environment variable.

## Configure and build

### Configure on Linux

```shell
# Assuming in IREE source root
$ cmake -G Ninja -B build-android  \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-29 \
    -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_HOST_C_COMPILER=`which clang` -DIREE_HOST_CXX_COMPILER=`which clang++`
```

*   The above configures IREE to cross-compile towards 64-bit
    (`-DANDROID_ABI="arm64-v8a"`) Android 10 (`-DANDROID_PLATFORM=android-29`).
    This may require the latest Android NDK release. You can choose the suitable
    [`ANDROID_ABI`](https://developer.android.com/ndk/guides/cmake#android_abi)
    and
    [`ANDROID_PLATFORM`](https://en.wikipedia.org/wiki/Android_version_history)
    for your target device. You can also refer to Android NDK's
    [CMake documentation](https://developer.android.com/ndk/guides/cmake) for
    more toolchain arguments.
*   Building IREE compilers and samples for Android is not supported at the
    moment; they will be enabled soon.
*   We need to define `IREE_HOST_{C|CXX}_COMPILER` to Clang here because IREE
    does [not support](https://github.com/google/iree/issues/1269) GCC well at
    the moment.

### Configure on Windows

On Windows, we will need the full path to the `cl.exe` compiler. This can be
obtained by
[opening a developer command prompt window](https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=vs-2019#developer_command_prompt)
and type `where cl.exe`. Then in a command prompt (`cmd.exe`):

```cmd
REM Assuming in IREE source root
> cmake -G Ninja -B build-android  \
    -DCMAKE_TOOLCHAIN_FILE="%ANDROID_NDK%/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-29 \
    -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_HOST_C_COMPILER="<full-path-to-cl.exe>" \
    -DIREE_HOST_CXX_COMPILER="<full-path-to-cl.exe>" \
    -DLLVM_HOST_TRIPLE="x86_64-pc-windows-msvc"
```

*   See the Linux section in the above for explanations of the used arguments.
*   We need to define `LLVM_HOST_TRIPLE` in the above because LLVM cannot
    properly detect host triple under Android CMake toolchain file. This might
    be fixed later.

### Build all targets

```shell
$ cmake --build build-android/
```

## Test on Android

Make sure you
[enable developer options and USB debugging](https://developer.android.com/studio/debug/dev-options#enable)
for your Android device.

Connect your Android device to the development machine and make sure you can see
the device when:

```shell
$ adb devices

List of devices attached
XXXXXXXXXXX     device
```

Then you can run all device tests via

```shell
$ cd build-android
$ ctest --output-on-failure
```

The above command will upload necessary build artifacts to the Android device's
`/data/local/tmp` directory, run the tests there, and report status back.

Alternatively, if you want to invoke a specific HAL backend on a IREE module:

### VMLA HAL backend

Translate a source MLIR into IREE module:

```shell
# Assuming in IREE source root
$ build-android/host/bin/iree-translate \
    -iree-mlir-to-vm-bytecode-module \
    -iree-hal-target-backends=vmla \
    iree/tools/test/simple.mlir \
    -o /tmp/simple-vmla.vmfb
```

Then push the IREE runtime executable and module to the device:

```shell
$ adb push build-android/iree/tools/iree-run-module /data/local/tmp/
$ adb shell chmod +x /data/local/tmp/iree-run-module
$ adb push /tmp/simple-vmla.vmfb /data/local/tmp/
```

Log into Android:

```shell
$ adb shell

android $ cd /data/local/tmp/
android $ ./iree-run-module -driver=vmla \
          -input_file=simple-vmla.vmfb \
          -entry_function=abs \
          -inputs="i32=-5"

EXEC @abs
i32=5
```

### Vulkan HAL backend

Please make sure your Android device is Vulkan capable. Vulkan is supported on
Android since 7, but Android 10 is our primary target at the moment.

Translate a source MLIR into IREE module:

```shell
# Assuming in IREE source root
$ build-android/host/bin/iree-translate \
    -iree-mlir-to-vm-bytecode-module \
    -iree-hal-target-backends=vulkan-spirv \
    iree/tools/test/simple.mlir \
    -o /tmp/simple-vulkan.vmfb
```

Then push the IREE runtime executable and module to the device:

```shell
$ adb push build-android/iree/tools/iree-run-module /data/local/tmp/
$ adb shell chmod +x /data/local/tmp/iree-run-module
$ adb push /tmp/simple-vulkan.vmfb /data/local/tmp/
```

Log into Android:

```shell
$ adb shell

android $ cd /data/local/tmp/
android $ ./iree-run-module -driver=vulkan \
          -input_file=simple-vulkan.vmfb \
          -entry_function=abs \
          -inputs="i32=-5"

EXEC @abs
i32=5
```

#### Common issues

##### Vulkan function `vkCreateInstance` not available

Since Android 8 Oreo, Android re-architected the OS framework with
[project Treble](https://source.android.com/devices/architecture#hidl).
Framework libraries and
[vendor libraries](https://source.android.com/devices/architecture/vndk) have a
more strict and clear separation. Their dependencies are carefully scrutinized
and only selected cases are allowed. This is enforced with
[linker namespaces](https://source.android.com/devices/architecture/vndk/linker-namespace).

`/data/local/tmp` is the preferred directory for automating native binary tests
built using NDK toolchain. They should be allowed to access libraries like
`libvulkan.so` for their functionality. However, there was an issue with fully
treblized Android 10 where `/data/local/tmp` did not have access to the linker
namespaces needed by `libvulkan.so`. This should be
[fixed](https://android.googlesource.com/platform/system/linkerconfig/+/296da5b1eb88a3527ee76352c2d987f82f3252eb)
now. But as typically in the Android system, it takes a long time to see the fix
getting propagated, if ever.

A known workaround is to symlink the vendor Vulkan implementation under
`/vendor/lib[64]` as `libvulkan.so` under `/data/local/tmp` and use
`LD_LIBRARY_PATH=/data/local/tmp` when invoking IREE executables.

For Qualcomm Adreno GPUs, the vendor Vulkan implemenation is at
`/vendor/lib[64]/hw/vulkan.*.so`. So for example for Snapdragon 865:

```shell
$ adb shell ln -s /vendor/lib64/hw/vulkan.kona.so /data/local/tmp/libvulkan.so
```

For ARM Mali GPUs, there is only one monolithic driver
(`/vendor/lib[64]/libGLES_mali.so`) for OpenGL and Vulkan and the Vulkan vendor
driver (`/vendor/lib[64]/hw/vulkan.*.so`) is just a symlink to it. So for
example:

```shell
$ adb shell ln -s /vendor/lib64/libGLES_mali.so /data/local/tmp/libvulkan.so
```

### Dylib LLVM AOT backend

To compile IREE module for the target Android device (assume Android 10 AArc64)
we need install the corresponding standalone toolchain and setting AOT linker
path environment variable:

```shell
$ export ANDROID_ARM64_TOOLCHAIN=/path/to/install/the/toolchain
$ $ANDROID_NDK/build/tools/make-standalone-toolchain.sh --arch=arm64 --platform=android-29 \
    --install-dir=$ANDROID_ARM64_TOOLCHAIN
$ export IREE_LLVMAOT_LINKER_PATH=$ANDROID_ARM64_TOOLCHAIN/aarch64-linux-android/bin/ld
```

Translate a source MLIR into an IREE module:

```shell
# Assuming in IREE source root
$ build-android/host/bin/iree-translate \
    -iree-mlir-to-vm-bytecode-module \
    -iree-llvm-target-triple=aarch64-linux-android \
    -iree-hal-target-backends=dylib-llvm-aot \
    iree/tools/test/simple.mlir \
    -o /tmp/simple-llvm_aot.vmfb
```

Then push the IREE runtime executable and module to the device:

```shell
$ adb push build-android/iree/tools/iree-run-module /data/local/tmp/
$ adb shell chmod +x /data/local/tmp/iree-run-module
$ adb push /tmp/simple-llvm_aot.vmfb /data/local/tmp/
```

Log into Android:

```shell
$ adb shell

android $ cd /data/local/tmp/
android $ ./iree-run-module -driver=dylib \
          -input_file=simple-llvm_aot.vmfb \
          -entry_function=abs \
          -inputs="i32=-5"

EXEC @abs
i32=5
```
