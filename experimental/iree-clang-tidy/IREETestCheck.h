// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CLANG_TIDY_TEST_CHECK_H_
#define IREE_CLANG_TIDY_TEST_CHECK_H_

#include <map>
#include <string>
#include <vector>

#include "ClangTidyCheck.h"

namespace clang::tidy::iree {

// `iree-test` check.
// This is here to test that the clang-tidy plugin functions and acts as an
// example for a check that takes options of various kinds. This can be used
// as a template for new checks.
struct IREETestCheck : public ClangTidyCheck {
  IREETestCheck(StringRef Name, ClangTidyContext *Context);
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;

  std::string TestString;
  std::vector<std::string> TestList;
  std::map<std::string, std::string> TestMap;
  int TestNumber;
  bool TestBool;
  std::string Severity;  // "warning", "error", or "remark"
};

}  // namespace clang::tidy::iree

#endif  // IREE_CLANG_TIDY_TEST_CHECK_H_
