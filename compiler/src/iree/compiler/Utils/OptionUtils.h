// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_FLAG_UTILS_H
#define IREE_COMPILER_UTILS_FLAG_UTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler {

// Base class that can bind named options to fields of structs.
//
// Typically use by adding the following to your struct:
//   void bindOptions(OptionsBinder &binder);
//   using FromFlags = OptionsFromFlags<MyStruct>;
//
// Then you can get the struct as initialized from global CL options as:
//   MyStruct::FromFlags::get()
//
// Such use is referred to as a "global binder". You can also create a
// local binder, which does not interact with global flags by calling the
// local() static factory. When in this mode, the lifetime of any bound
// structures must exceed uses of the binder to parse or print (ie. via
// parseArguments() or printArguments()).
//
// The underlying LLVM command line support is quite flexible and all esoteric
// features are not supported here. Consider that supported structs can define
// options of built-in scalar types (string, ints, bool, etc) and enums. Lists
// of built-in scalar types are also supported.
class OptionsBinder {
public:
  static OptionsBinder global() { return OptionsBinder(); }

  static OptionsBinder local() {
    return OptionsBinder(std::make_unique<llvm::cl::SubCommand>());
  }

  template <typename T, typename V, typename... Mods>
  void opt(llvm::StringRef name, V &value, Mods... Ms) {
    if (!scope) {
      // Bind global options.
      auto opt = std::make_unique<llvm::cl::opt<T, /*ExternalStorage=*/true>>(
          name, llvm::cl::location(value), llvm::cl::init(value),
          std::forward<Mods>(Ms)...);
      addGlobalOption(std::move(opt));
    } else {
      // Bind local options.
      auto option =
          std::make_unique<llvm::cl::opt<T, /*ExternalStorage=*/true>>(
              name, llvm::cl::sub(*scope), llvm::cl::location(value),
              llvm::cl::init(value), std::forward<Mods>(Ms)...);
      auto printCallback =
          makePrintCallback(option->ArgStr, option->getParser(), &value);
      auto changedCallback = makeChangedCallback(&value);
      localOptions.push_back(
          LocalOptionInfo{std::move(option), printCallback, changedCallback});
    }
  }

  template <typename T, typename V, typename... Mods>
  void list(llvm::StringRef name, V &value, Mods... Ms) {
    if (!scope) {
      // Bind global options.
      auto list =
          std::make_unique<llvm::cl::list<T>>(name, std::forward<Mods>(Ms)...);
      // Since list does not support external storage, hook the callback
      // and use it to update.
      list->setCallback(
          [&value](const T &newElement) { value.push_back(newElement); });
      addGlobalOption(std::move(list));
    } else {
      // Bind local options.
      auto list = std::make_unique<llvm::cl::list<T>>(
          name, llvm::cl::sub(*scope), std::forward<Mods>(Ms)...);
      auto printCallback =
          makeListPrintCallback(list->ArgStr, list->getParser(), &value);
      auto changedCallback = makeListChangedCallback(&value);
      // Since list does not support external storage, hook the callback
      // and use it to update.
      list->setCallback(
          [&value](const T &newElement) { value.push_back(newElement); });

      localOptions.push_back(
          LocalOptionInfo{std::move(list), printCallback, changedCallback});
    }
  }

  // For a local binder, parses a sequence of flags of the usual form on
  // command lines.
  using ErrorCallback = std::function<void(llvm::StringRef message)>;
  LogicalResult parseArguments(int argc, const char *const *argv,
                               ErrorCallback onError = nullptr);

  // Prints any flag values that differ from their default.
  // Flags print in the order declared, which preserves some notion of grouping
  // and is stable.
  llvm::SmallVector<std::string> printArguments(bool nonDefaultOnly = false);

private:
  struct LocalOptionInfo {
    using ChangedCallback = std::function<bool()>;
    using PrintCallback = std::function<void(llvm::raw_ostream &)>;
    std::unique_ptr<llvm::cl::Option> option;
    PrintCallback print;
    ChangedCallback isChanged;
  };

  OptionsBinder() = default;
  OptionsBinder(std::unique_ptr<llvm::cl::SubCommand> scope)
      : scope(std::move(scope)) {}
  void addGlobalOption(std::unique_ptr<llvm::cl::Option> option);

  // LLVM makes a half-hearted (i.e. "best effort" == "no effort") attempt to
  // handle non-enumerated generic value based options, but the generic
  // comparisons are not reliably implemented. Simplify our lives by only
  // supporting int convertible values (i.e. enums), which we can restrict
  // ourselves to.
  // Scalar enum print specialization.
  template <typename V, typename ParserTy>
  static auto makePrintCallback(llvm::StringRef optionName, ParserTy &parser,
                                V *value)
      -> decltype(static_cast<llvm::cl::generic_parser_base &>(parser),
                  static_cast<int>(*value), LocalOptionInfo::PrintCallback()) {
    return [optionName, &parser, value](llvm::raw_ostream &os) {
      StringRef valueName("<unknown>");
      for (unsigned i = 0; i < parser.getNumOptions(); ++i) {
        V cmpValue = static_cast<const llvm::cl::OptionValue<V> &>(
                         parser.getOptionValue(i))
                         .getValue();
        if (cmpValue == *value) {
          valueName = parser.getOption(i);
          break;
        }
      }
      os << "--" << optionName << "=" << valueName;
    };
  }

  // Basic scalar print specialization.
  template <typename V, typename ParserTy>
  static auto makePrintCallback(llvm::StringRef optionName, ParserTy &parser,
                                V *value)
      -> decltype(static_cast<llvm::cl::basic_parser<V> &>(parser),
                  LocalOptionInfo::PrintCallback()) {
    return [optionName, value](llvm::raw_ostream &os) {
      os << "--" << optionName << "=" << *value;
    };
  }

  // Bool scalar print specialization.
  template <typename ParserTy>
  static auto makePrintCallback(llvm::StringRef optionName, ParserTy &parser,
                                bool *value)
      -> decltype(static_cast<llvm::cl::basic_parser<bool> &>(parser),
                  LocalOptionInfo::PrintCallback()) {
    return [optionName, value](llvm::raw_ostream &os) {
      os << "--" << optionName << "=";
      if (*value) {
        os << "true";
      } else {
        os << "false";
      }
    };
  }

  // Scalar changed specialization.
  template <typename V>
  static LocalOptionInfo::ChangedCallback makeChangedCallback(V *currentValue) {
    // Capture the current value as the initial value.
    V initialValue = *currentValue;
    return [currentValue, initialValue]() -> bool {
      return *currentValue != initialValue;
    };
  }

  // List changed specialization.
  template <typename V>
  static LocalOptionInfo::ChangedCallback
  makeListChangedCallback(V *currentValue) {
    return [currentValue]() -> bool { return !currentValue->empty(); };
  }

  // List basic print specialization. This is the only one we provide so far.
  // Add others if the compiler tells you to.
  template <typename ListTy, typename ParserTy,
            typename V = typename ListTy::value_type>
  static auto makeListPrintCallback(llvm::StringRef optionName,
                                    ParserTy &parser, ListTy *values)
      -> decltype(static_cast<llvm::cl::basic_parser<V> &>(parser),
                  LocalOptionInfo::PrintCallback()) {
    return [optionName, values](llvm::raw_ostream &os) {
      os << "--" << optionName << "=";
      for (auto it : llvm::enumerate(*values)) {
        if (it.index() > 0)
          os << ",";
        os << it.value();
      }
    };
  }

  std::unique_ptr<llvm::cl::SubCommand> scope;
  llvm::SmallVector<LocalOptionInfo> localOptions;
};

// Generic class that is used for allocating an Options class that initializes
// from flags. Every Options type that can have FromFlags called on it needs
// to include definitions in one implementation module (at the top level
// namespace):
//   IREE_DEFINE_COMPILER_OPTION_FLAGS(DerivedTy);
template <typename DerivedTy>
class OptionsFromFlags {
public:
  static DerivedTy &get();
};

#define IREE_DEFINE_COMPILER_OPTION_FLAGS(DerivedTy)                           \
  template <>                                                                  \
  DerivedTy &mlir::iree_compiler::OptionsFromFlags<DerivedTy>::get() {         \
    struct InitializedTy : DerivedTy {                                         \
      InitializedTy() {                                                        \
        mlir::iree_compiler::OptionsBinder binder =                            \
            mlir::iree_compiler::OptionsBinder::global();                      \
        DerivedTy::bindOptions(binder);                                        \
      }                                                                        \
    };                                                                         \
    static InitializedTy singleton;                                            \
    return singleton;                                                          \
  }

} // namespace mlir::iree_compiler

namespace llvm {
namespace cl {

struct ByteSize {
  int64_t value = 0;
  ByteSize() = default;
  ByteSize(int64_t value) : value(value) {}
  operator bool() const noexcept { return value != 0; }
};

struct PowerOf2ByteSize : public ByteSize {
  using ByteSize::ByteSize;
};

extern template class basic_parser<ByteSize>;
extern template class basic_parser<PowerOf2ByteSize>;

template <>
class parser<ByteSize> : public basic_parser<ByteSize> {
public:
  parser(Option &O) : basic_parser(O) {}
  bool parse(Option &O, StringRef ArgName, StringRef Arg, ByteSize &Val);
  StringRef getValueName() const override { return "byte size"; }
  void printOptionDiff(const Option &O, ByteSize V, const OptVal &Default,
                       size_t GlobalWidth) const;
  void anchor() override;
};

template <>
class parser<PowerOf2ByteSize> : public basic_parser<PowerOf2ByteSize> {
public:
  parser(Option &O) : basic_parser(O) {}
  bool parse(Option &O, StringRef ArgName, StringRef Arg,
             PowerOf2ByteSize &Val);
  StringRef getValueName() const override { return "power of two byte size"; }
  void printOptionDiff(const Option &O, PowerOf2ByteSize V,
                       const OptVal &Default, size_t GlobalWidth) const;
  void anchor() override;
};

} // namespace cl
} // namespace llvm

#endif // IREE_COMPILER_UTILS_FLAG_UTILS_H
