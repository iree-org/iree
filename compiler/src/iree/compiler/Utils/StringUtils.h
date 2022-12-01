// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_STRINGUTILS_H_
#define IREE_COMPILER_UTILS_STRINGUTILS_H_

#include <string>

#include "mlir/Support/LLVM.h"

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

// Sanitizes a symbol name for compatibility with common targets (C, file
// systems, debug databases, etc). Characters outside the allowed range are
// replaced by `_`.
//
// If coalesce_delims is true then consecutive `_` are replaced by a single `_`.
//
// MLIR identifiers must match this regex:
//   (letter|[_]) (letter|digit|[_$.])*
// https://mlir.llvm.org/docs/LangRef/#identifiers-and-keywords
// This is a superset of the names other targets support and as inputs are only
// expected to match the above any place exporting symbol names must use this.
//
// Examples:
//  `abc` -> `abc`
//  `a.b` -> `a_b`
//  `a$-æb` -> `a___b`
//
// If coalesce_delims is true then:
//  `a$-æb` -> `a_b`
std::string sanitizeSymbolName(StringRef name, bool coalesce_delims = false);

// Sanitizes a file name for compatibility with common file systems.
//
// Examples:
//  `abc` -> `abc`
//  `a.b` -> `a.b`
//  `a$-æb` -> `a_-_b`
std::string sanitizeFileName(StringRef name);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_STRINGUTILS_H_
