---
hide:
  - tags
tags:
  - iOS
---

# iOS cross-compilation

Cross-compilation for iOS consists of the two steps below.

* On the macOS host, build the IREE compiler.  We can run it to create
  IREE modules.
* Build the IREE runtime on the macOS host for iOS devices and the
  simulator.  We can then run the IREE module on the simulator.

## Prerequisites

### Install Xcode and iOS SDK

For cross-compilation, you need
[Xcode](https://developer.apple.com/xcode/). It comes with the SDKs
for iOS devices and the simulator, as well as the `simctl` tool for
controlling the simulator from the command line.

### Host environment setup

On your host platform, you should already be able to build IREE from
source.  Please make sure you've gone through the steps in [getting
started](./getting-started.md).

## Configure and Build

### Build the IREE Compiler for the Host

Build and install on your macOS host:

``` shell
cmake -S . -B ../iree-build/ -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=../iree-build/install

cmake --build ../iree-build/ --target install
```

## Cross-compile the IREE Runtime for iOS

Build the runtime for the iOS Simulator.

``` shell
cmake -S . -B ../build-ios-sim -GNinja \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=$(xcodebuild -version -sdk iphonesimulator Path) \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_SYSTEM_PROCESSOR=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DCMAKE_IOS_INSTALL_COMBINED=YES \
  -DIREE_HOST_BIN_DIR="$PWD/../iree-build/install/bin" \
  -DCMAKE_INSTALL_PREFIX=../build-ios-sim/install \
  -DIREE_BUILD_COMPILER=OFF

cmake --build ../build-ios-sim --config Release --target install
```

Or, we can build the runtime for iOS devices it by changing the value
of the `-DCMAKE OSX SYSROOT` option to:

``` shell
  -DCMAKE_OSX_SYSROOT=$(xcodebuild -version -sdk iphoneos Path)
```

## Running IREE Modules on the iOS Simulator

Run the IREE compiler on the host to generate a module.

``` shell
../iree-build/install/bin/iree-compile \
  --iree-hal-target-backends=vmvx \
  samples/models/simple_abs.mlir \
  -o /tmp/simple_abs_vmvx.vmfb
```

We could test the generated module by running the macOS version of
`iree-run-module` on the host.

``` shell
../iree-build/install/bin/iree-run-module \
  --module=/tmp/simple_abs_vmvx.vmfb \
  --device=local-task \
  --function=abs \
  --input="f32=-5"
```

To run it on the iOS simulator, we need to copy the vmfb file into the
`iree-run-module` iOS app bundle.

``` shell
cp /tmp/simple_abs_vmvx.vmfb \
   ../build-ios-sim/install/bin/iree-run-module.app/
```

Open the iOS Simulator Manager on the host.

``` shell
open -a Simulator
```

After creating and booting a simulator in this app, you can list it
from the command-line.

``` shell
xcrun simctl list devices | grep Booted
```

This is what should come out of the command:

``` text
    iPhone 14 Pro (12341234-ABCD-ABCD-ABCD-123412341234) (Booted)
```

where `iPhone 14 Pro` is the device being simulated and
`12341234-ABCD-ABCD-ABCD-123412341234` is the simulator's _unique
device ID_ (UDID).

Install the app `iree-run-module` on the simulator, given its UDID.

``` shell
xcrun simctl install <UDID> ../build-ios-sim/install/bin/iree-run-module.app
```

Check the path to the installed bundle, where the
`simple_abs_vmvx.vmfb` module should be found.

``` shell
ls $(xcrun simctl get_app_container <UDID> dev.iree.iree-run-module)
```

The string `dev.iree.iree-run-module` is the _bundle identifier_ of
the iOS app.  The CMake building process generates it and saves it in
the _property list_ (plist) file
`../build-ios-sim/install/bin/iree-run-module.app/Info.plist`.

Launch the `iree-run-module` app on the simulator to run the IREE
module `simple_abs_vmvx.vmfb`.

``` shell
xcrun simctl launch --console \
  <UDID> \
  dev.iree.runmodule \
  --device=local-task \
  --function=abs \
  --input="f32=-5" \
  --module=$(xcrun simctl get_app_container <UDID> dev.iree.iree-run-module)/simple_abs_vmvx.vmfb
```
