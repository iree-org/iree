// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===----------------------------------------------------------------------===//
//
//         ██     ██  █████  ██████  ███    ██ ██ ███    ██  ██████
//         ██     ██ ██   ██ ██   ██ ████   ██ ██ ████   ██ ██
//         ██  █  ██ ███████ ██████  ██ ██  ██ ██ ██ ██  ██ ██   ███
//         ██ ███ ██ ██   ██ ██   ██ ██  ██ ██ ██ ██  ██ ██ ██    ██
//          ███ ███  ██   ██ ██   ██ ██   ████ ██ ██   ████  ██████
//
//===----------------------------------------------------------------------===//
//
// This file controls global configuration parameters used throughout IREE.
// Each option added here should be considered something worth enabling an
// entirely new testing configuration to test and may involve fanning out many
// configurations depending on which flags are mutually non-exclusive.
// Err on the side of using runtime flags for options that have minimal impact
// to code size or toolchain requirements of our more constrained targets.
//
// Examples of good configuration settings:
// - remote HAL device pointer size (cannot be inferred from local config)
// - no-op override on synchronization primitives (unsafe, untested)
//
// Examples of bad configuration settings:
// - which HAL backend to use (better as build configuration; link what you use)

#ifndef IREE_BASE_CONFIG_H_
#define IREE_BASE_CONFIG_H_

#include "iree/base/target_platform.h"

//===----------------------------------------------------------------------===//
// User configuration overrides
//===----------------------------------------------------------------------===//
// A user include file always included prior to any IREE configuration. This is
// used to override the default configuration in this file without needing to
// modify the IREE code.
//
// Specify a custom file with `-DIREE_USER_CONFIG_H="my_config.h"`.

#if defined(IREE_USER_CONFIG_H)
#include IREE_USER_CONFIG_H
#endif  // IREE_USER_CONFIG_H

//===----------------------------------------------------------------------===//
// Pointer size specification
//===----------------------------------------------------------------------===//
// IREE uses two pointer classes throughout its code:
//
//  `iree_host_size_t`:
//    The native pointer size of the local "host" code. This is always C's
//    size_t but is aliased to make it easier to differentiate from
//    "unspecified" size_t and iree_device_size_t. Always prefer using this for
//    sizes of pointers that never leave the host.
//
//  `iree_device_size_t`:
//    The pointer size - possibly larger than needed - for remote "device" code.
//    As the host and device may be running on entirely different machines it is
//    often best to use a conservative value for this: a 32-bit host may be
//    submitting work for a 64-bit device, and using a 32-bit size_t for device
//    pointers would truncate bits and prevent round-tripping.
//
// The specific values for these can be overridden with configuration settings:

#if !defined(IREE_HOST_SIZE_T)
#define IREE_HOST_SIZE_T size_t
#endif  // !IREE_HOST_SIZE_T

// Size, in bytes, of a buffer on the local host.
typedef IREE_HOST_SIZE_T iree_host_size_t;

#if !defined(IREE_DEVICE_SIZE_T)
#define IREE_DEVICE_SIZE_T uint64_t
#endif  // !IREE_DEVICE_SIZE_T

// Size, in bytes, of a buffer on remote devices.
typedef IREE_DEVICE_SIZE_T iree_device_size_t;

//===----------------------------------------------------------------------===//
// Synchronization and threading
//===----------------------------------------------------------------------===//
// On ultra-tiny systems where there may only be a single core - or a single
// core that is guaranteed to ever call an IREE API - all synchronization
// primitives used throughout IREE can be turned into no-ops. Note that behavior
// is undefined if there is use of any `iree_*` API call or memory that is
// owned by IREE. Unless your target system is in a similar class to an Arduino
// this is definitely not what you want.

#if !defined(IREE_SYNCHRONIZATION_DISABLE_UNSAFE)
#define IREE_SYNCHRONIZATION_DISABLE_UNSAFE 0
#endif  // !IREE_SYNCHRONIZATION_DISABLE_UNSAFE

//===----------------------------------------------------------------------===//
// IREE VM configuration
//===----------------------------------------------------------------------===//
// Enables optional VM features. Each of these adds a few KB to the final binary
// when using the IREE VM. The compiler must be configured to the same set of
// available extensions in order to ensure that the compiled modules only use
// features available on the target they are to run on.
//
// See the `-iree-vm-target-extensions=` compiler option for more information.

#if !defined(IREE_VM_EXT_I64_ENABLE)
// Enables the 64-bit integer instruction extension.
// Targeted from the compiler with `-iree-vm-target-extensions=i64`.
#define IREE_VM_EXT_I64_ENABLE 1
#endif  // !IREE_VM_EXT_I64_ENABLE

#if !defined(IREE_VM_EXT_F32_ENABLE)
// Enables the 32-bit floating-point instruction extension.
// Targeted from the compiler with `-iree-vm-target-extensions=f32`.
#define IREE_VM_EXT_F32_ENABLE 1
#endif  // !IREE_VM_EXT_F32_ENABLE

#if !defined(IREE_VM_EXT_F64_ENABLE)
// Enables the 64-bit floating-point instruction extension.
// Targeted from the compiler with `-iree-vm-target-extensions=f64`.
#define IREE_VM_EXT_F64_ENABLE 0
#endif  // !IREE_VM_EXT_F64_ENABLE

#endif  // IREE_BASE_CONFIG_H_
