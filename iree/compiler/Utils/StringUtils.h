// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_STRINGUTILS_H_
#define IREE_COMPILER_UTILS_STRINGUTILS_H_

#include <string>

namespace mlir {
namespace iree_compiler {

// Replaces all occurrences of `match` in `str` with `substitute`.
// Operates in place.
void replaceAllSubstrsInPlace(std::string &str, const std::string &match,
                              const std::string &substitute);

// Replaces all occurrences of `match` in `str` with `substitute`.
// Does not mutate its arguments, returns the new string.
std::string replaceAllSubstrs(const std::string &str, const std::string &match,
                              const std::string &substitute);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_STRINGUTILS_H_
