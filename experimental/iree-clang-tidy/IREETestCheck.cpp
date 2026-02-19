// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "IREETestCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace clang::ast_matchers;

namespace clang::tidy::iree {

IREETestCheck::IREETestCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      TestString(Options.get("TestString", "")),
      TestNumber(Options.get("TestNumber", 0)),
      TestBool(Options.get("TestBool", false)),
      Severity(Options.get("Severity", "warning")) {
  // Parse TestList as comma-separated values.
  StringRef ListStr = Options.get("TestList", "");
  if (!ListStr.empty()) {
    SmallVector<StringRef, 4> Parts;
    ListStr.split(Parts, ',');
    for (StringRef Part : Parts) {
      TestList.push_back(Part.trim().str());
    }
  }

  // Parse TestMap as comma-separated key:value pairs.
  StringRef MapStr = Options.get("TestMap", "");
  if (!MapStr.empty()) {
    SmallVector<StringRef, 4> Pairs;
    MapStr.split(Pairs, ',');
    for (StringRef Pair : Pairs) {
      size_t ColonPos = Pair.find(':');
      if (ColonPos != StringRef::npos) {
        StringRef Key = Pair.substr(0, ColonPos).trim();
        StringRef Value = Pair.substr(ColonPos + 1).trim();
        TestMap[Key.str()] = Value.str();
      }
    }
  }
}

void IREETestCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "TestString", TestString);
  Options.store(Opts, "TestNumber", TestNumber);
  Options.store(Opts, "TestBool", TestBool);
  Options.store(Opts, "Severity", Severity);

  // Store TestList as comma-separated string.
  std::string ListStr;
  for (size_t i = 0; i < TestList.size(); ++i) {
    if (i > 0) ListStr += ",";
    ListStr += TestList[i];
  }
  Options.store(Opts, "TestList", ListStr);

  // Store TestMap as comma-separated key:value pairs.
  std::string MapStr;
  bool First = true;
  for (const auto &KV : TestMap) {
    if (!First) MapStr += ",";
    First = false;
    MapStr += KV.first + ":" + KV.second;
  }
  Options.store(Opts, "TestMap", MapStr);
}

void IREETestCheck::registerMatchers(MatchFinder *Finder) {
  // Match function declarations named "test_iree_check".
  Finder->addMatcher(
      functionDecl(hasName("test_iree_check")).bind("test_function"), this);
}

void IREETestCheck::check(const MatchFinder::MatchResult &Result) {
  // Check for test function.
  if (const auto *FuncDecl =
          Result.Nodes.getNodeAs<FunctionDecl>("test_function")) {
    // Build a diagnostic message that includes configuration values for
    // testing.
    std::string Msg = "found test function 'test_iree_check'";

    // Add TestString if configured.
    if (!TestString.empty()) {
      Msg += " [TestString=" + TestString + "]";
    }

    // Add TestNumber if non-zero.
    if (TestNumber != 0) {
      Msg += " [TestNumber=" + std::to_string(TestNumber) + "]";
    }

    // Add TestBool.
    Msg += " [TestBool=" + std::string(TestBool ? "true" : "false") + "]";

    // Add TestList if not empty.
    if (!TestList.empty()) {
      Msg += " [TestList=";
      for (size_t i = 0; i < TestList.size(); ++i) {
        if (i > 0) Msg += ",";
        Msg += TestList[i];
      }
      Msg += "]";
    }

    // Add TestMap if not empty.
    if (!TestMap.empty()) {
      Msg += " [TestMap=";
      bool First = true;
      for (const auto &KV : TestMap) {
        if (!First) Msg += ",";
        First = false;
        Msg += KV.first + ":" + KV.second;
      }
      Msg += "]";
    }

    // Use configured severity level.
    DiagnosticIDs::Level DiagLevel = DiagnosticIDs::Warning;
    if (Severity == "error") {
      DiagLevel = DiagnosticIDs::Error;
    } else if (Severity == "remark") {
      DiagLevel = DiagnosticIDs::Remark;
    }

    diag(FuncDecl->getLocation(), Msg, DiagLevel) << FuncDecl->getSourceRange();
  }
}

}  // namespace clang::tidy::iree
