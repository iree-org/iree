// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/temp_file.h"

#include <cstdint>
#include <random>
#include <string>

#include "iree/testing/gtest.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#else
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_WINDOWS

namespace iree::testing {
namespace {

static bool IsPathSeparator(char c) { return c == '/' || c == '\\'; }

static bool IsDirectory(const std::string& path) {
#if defined(IREE_PLATFORM_WINDOWS)
  const DWORD attributes = GetFileAttributesA(path.c_str());
  return attributes != INVALID_FILE_ATTRIBUTES &&
         (attributes & FILE_ATTRIBUTE_DIRECTORY);
#else
  struct stat stat_buf;
  return stat(path.c_str(), &stat_buf) == 0 && S_ISDIR(stat_buf.st_mode);
#endif  // IREE_PLATFORM_WINDOWS
}

static iree_status_t CreateSingleDirectory(const std::string& path) {
  if (path.empty() || IsDirectory(path)) return iree_ok_status();
#if defined(IREE_PLATFORM_WINDOWS)
  if (CreateDirectoryA(path.c_str(), NULL)) return iree_ok_status();
  const DWORD error = GetLastError();
  if (error == ERROR_ALREADY_EXISTS && IsDirectory(path)) {
    return iree_ok_status();
  }
  return iree_make_status(iree_status_code_from_win32_error(error),
                          "failed to create test temp directory '%s'",
                          path.c_str());
#else
  if (mkdir(path.c_str(), 0777) == 0) return iree_ok_status();
  const int saved_errno = errno;
  if (saved_errno == EEXIST && IsDirectory(path)) return iree_ok_status();
  return iree_make_status(iree_status_code_from_errno(saved_errno),
                          "failed to create test temp directory '%s'",
                          path.c_str());
#endif  // IREE_PLATFORM_WINDOWS
}

static iree_status_t EnsureDirectoryExists(std::string path) {
  if (path.empty()) return iree_ok_status();
  while (path.size() > 1 && IsPathSeparator(path.back())) {
    path.pop_back();
  }

  size_t first_component_position = 0;
#if defined(IREE_PLATFORM_WINDOWS)
  if (path.size() >= 2 && path[1] == ':') {
    first_component_position =
        path.size() >= 3 && IsPathSeparator(path[2]) ? 3 : 2;
  } else if (path.size() >= 2 && IsPathSeparator(path[0]) &&
             IsPathSeparator(path[1])) {
    const size_t server_end = path.find_first_of("/\\", 2);
    if (server_end == std::string::npos) return iree_ok_status();
    const size_t share_end = path.find_first_of("/\\", server_end + 1);
    first_component_position =
        share_end == std::string::npos ? path.size() : share_end + 1;
  }
#endif  // IREE_PLATFORM_WINDOWS

  for (size_t i = first_component_position; i <= path.size(); ++i) {
    if (i != path.size() && !IsPathSeparator(path[i])) continue;
    if (i == 0) continue;
    std::string prefix = path.substr(0, i);
    if (prefix.empty() || prefix.back() == ':') continue;
    IREE_RETURN_IF_ERROR(CreateSingleDirectory(prefix));
  }
  return iree_ok_status();
}

}  // namespace

std::string MakeTempFilePath(const char* stem, const char* suffix) {
  IREE_CHECK_OK(EnsureDirectoryExists(::testing::TempDir()));
  std::random_device random_device;
  const uint64_t random_value =
      (static_cast<uint64_t>(random_device()) << 32) | random_device();
  return ::testing::TempDir() + stem + "_" + std::to_string(random_value) +
         suffix;
}

bool TempFilePath::Exists() const {
  if (path_.empty()) return false;
#if defined(IREE_PLATFORM_WINDOWS)
  return GetFileAttributesA(path_.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
  struct stat stat_buf;
  return stat(path_.c_str(), &stat_buf) == 0;
#endif  // IREE_PLATFORM_WINDOWS
}

bool TempFilePath::Remove() const {
  if (path_.empty()) return false;
#if defined(IREE_PLATFORM_WINDOWS)
  return DeleteFileA(path_.c_str()) != 0;
#else
  return unlink(path_.c_str()) == 0;
#endif  // IREE_PLATFORM_WINDOWS
}

}  // namespace iree::testing
