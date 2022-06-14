// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/StringUtils.h"

namespace mlir {
namespace iree_compiler {

void replaceAllSubstrsInPlace(std::string &str, const std::string &match,
                              const std::string &substitute) {
  std::string::size_type scanLoc = 0, matchLoc = std::string::npos;
  while ((matchLoc = str.find(match, scanLoc)) != std::string::npos) {
    str.replace(matchLoc, match.size(), substitute);
    scanLoc = matchLoc + substitute.size();
  }
}

std::string replaceAllSubstrs(const std::string &str, const std::string &match,
                              const std::string &substitute) {
  std::string copy(str);
  replaceAllSubstrsInPlace(copy, match, substitute);
  return copy;
}

}  // namespace iree_compiler
}  // namespace mlir
