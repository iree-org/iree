// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_EMBEDDEDDATADIRECTORY_H_
#define IREE_COMPILER_UTILS_EMBEDDEDDATADIRECTORY_H_

#include <mutex>
#include "llvm/ADT/StringMap.h"

namespace mlir::iree_compiler {

// A string-to-StringRef map that acts as a virtual filesystem: the keys are
// "filenames" and the values are file contents.
class EmbeddedDataDirectory {
public:
  // Calls the given `callback` on a global singleton object, guarded by a
  // global mutex.
  //
  // Only use this for use cases that require a global object, such as when
  // exporting data between parts of the compiler that can't directly link to
  // each other (e.g. from a plugin to outside of the plugin).
  static void
  withGlobal(llvm::function_ref<void(EmbeddedDataDirectory &)> callback) {
    static EmbeddedDataDirectory dir;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    callback(dir);
  }

  // Add a new entry if it didn't already exist. Return `true` if it was added.
  bool addFile(llvm::StringRef fileName, llvm::StringRef contents) {
    auto [_iter, success] = map.insert({fileName, contents});
    return success;
  }

  // Get an existing entry if it exists, otherwise return nullopt.
  std::optional<llvm::StringRef> getFile(llvm::StringRef fileName) const {
    auto iter = map.find(fileName);
    if (iter == map.end()) {
      return std::nullopt;
    }
    return iter->getValue();
  }

  // Direct access to the underlying StringMap, for use cases that are not well
  // served by convenience methods like addFile and getFile. For example,
  // iterating over all entries.
  llvm::StringMap<llvm::StringRef> &getMap() { return map; }

private:
  llvm::StringMap<llvm::StringRef> map;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_EMBEDDEDDATADIRECTORY_H_
