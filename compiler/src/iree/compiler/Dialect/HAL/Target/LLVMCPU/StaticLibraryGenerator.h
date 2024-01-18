// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_STATICLIBRARYGENERATOR_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_STATICLIBRARYGENERATOR_H_

#include <string>

namespace mlir::iree_compiler::IREE::HAL {

// Produces a static executable library and generated '.h'.
// The temporary object file is copied to the library_output_path. The '.h' file
// with the query_function_name is placed beside it (using the same base
// filename of the library). Returns true if successful.
bool outputStaticLibrary(const std::string &library_name,
                         const std::string &query_function_name,
                         const std::string &library_output_path,
                         const std::string &temp_object_path);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_LLVMCPU_STATICLIBRARYGENERATOR_H_
