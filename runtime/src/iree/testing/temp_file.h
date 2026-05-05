// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TESTING_TEMP_FILE_H_
#define IREE_TESTING_TEMP_FILE_H_

#include <string>

#include "iree/base/api.h"

namespace iree::testing {

// Returns a unique file path under the active test temp directory.
//
// The returned path is not created and is not deleted by this helper. The
// containing temp directory is created if the test runner has not already done
// so, while the runner still owns its lifetime and can preserve artifacts for
// debugging.
std::string MakeTempFilePath(const char* stem, const char* suffix = "");

// Move-only generated temp path string.
class TempFilePath {
 public:
  TempFilePath() = default;
  explicit TempFilePath(const char* stem, const char* suffix = "")
      : path_(MakeTempFilePath(stem, suffix)) {}

  TempFilePath(TempFilePath&&) noexcept = default;
  TempFilePath& operator=(TempFilePath&&) noexcept = default;

  TempFilePath(const TempFilePath&) = delete;
  TempFilePath& operator=(const TempFilePath&) = delete;

  explicit operator bool() const { return !path_.empty(); }

  const std::string& path() const { return path_; }

  iree_string_view_t path_view() const {
    return iree_make_string_view(path_.data(), path_.size());
  }

  // Returns true if the path currently names an existing file.
  bool Exists() const;

  // Removes the file at the generated path.
  bool Remove() const;

  void Reset() { path_.clear(); }

 private:
  // Generated path string.
  std::string path_;
};

}  // namespace iree::testing

#endif  // IREE_TESTING_TEMP_FILE_H_
