// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// numpy-compatible(ish) IO
//===----------------------------------------------------------------------===//
//
// Provides C variants of the numpy.load/numpy.save routines:
// https://numpy.org/doc/stable/reference/routines.io.html
// which uses the given file format:
// https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
//
// Pickled objects are not supported (similar to using `allow_pickle=False`) and
// not all dtypes are supported.
//
// .npy and uncompressed .npz files can be mapped into host memory with
// IREE_NUMPY_NPY_LOAD_OPTION_MAP_FILE if the HAL device allocator
// supports using such memory. On devices with discrete memory the contents will
// be loaded into host memory and copied to the device.
//
// This current implementation is very basic; in the future it'd be nice to
// support an iree_io_stream_t to allow for externalizing the file access.
//
// NOTE: this implementation is optimized for code size and is not intended to
// be used in performance-sensitive situations. If you are wanting to run
// through thousands of arrays and GB of data then you're going to want
// something more sophisticated (delay loading with async IO, etc).
//
// TODO(benvanik): conditionally enable compression when zlib is present. For
// now to reduce dependencies we don't support loading compressed npz files or
// compression on save.

#ifndef IREE_TOOLING_NUMPY_IO_H_
#define IREE_TOOLING_NUMPY_IO_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/stream.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// .npy (multiple values concatenated)
//===----------------------------------------------------------------------===//

// Options controlling npy load behavior.
enum iree_numpy_npy_load_options_bits_t {
  IREE_NUMPY_NPY_LOAD_OPTION_DEFAULT = 0u,

  // Tries to map the file into memory and use the contents directly from the
  // file system. Only available if the HAL device supports accessing mapped
  // data.
  // Like providing `mmap_mode` to `numpy.load`.
  // May be ignored if the implementation does not support mapping.
  IREE_NUMPY_NPY_LOAD_OPTION_MAP_FILE = 1u << 0,
};
typedef uint32_t iree_numpy_npy_load_options_t;

// Options controlling npy save behavior.
enum iree_numpy_npy_save_options_bits_t {
  IREE_NUMPY_NPY_SAVE_OPTION_DEFAULT = 0u,
};
typedef uint32_t iree_numpy_npy_save_options_t;

// Loads a single value from a .npy |stream| into a buffer view.
// On success |out_buffer_view| will have a buffer view matching the parameters
// in the npy file allocated from the given |device_allocator|.
//
// If IREE_NUMPY_NPY_LOAD_OPTION_MAP_FILE is set and the
// |device_allocator| supports mapping then the file will be mapped into the
// host process. Otherwise the file will be loaded into a new allocation.
//
// Upon return the |stream| will be positioned immediately following the
// ndarray contents, which may be end-of-stream.
// Fails if the contents of the npy cannot be mapped to IREE HAL types or is
// not an ndarray.
//
// See `numpy.load`:
// https://numpy.org/doc/stable/reference/generated/numpy.load.html
IREE_API_EXPORT iree_status_t iree_numpy_npy_load_ndarray(
    iree_io_stream_t* stream, iree_numpy_npy_load_options_t options,
    iree_hal_buffer_params_t buffer_params, iree_hal_device_t* device,
    iree_hal_allocator_t* device_allocator,
    iree_hal_buffer_view_t** out_buffer_view);

// Saves |buffer_view| to a .npy |stream|.
// The ndarray will be appended to the stream to produce a concatenated file.
//
// See `numpy.save`:
// https://numpy.org/doc/stable/reference/generated/numpy.save.html
IREE_API_EXPORT iree_status_t iree_numpy_npy_save_ndarray(
    iree_io_stream_t* stream, iree_numpy_npy_save_options_t options,
    iree_hal_buffer_view_t* buffer_view, iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_TOOLING_NUMPY_IO_H_
