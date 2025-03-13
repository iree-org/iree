// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_FLAG_UTILS_H
#define IREE_COMPILER_UTILS_FLAG_UTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
inline raw_ostream &operator<<(raw_ostream &os,
                               const llvm::OptimizationLevel &opt) {
  return os << 'O' << opt.getSpeedupLevel();
}
} // namespace llvm

namespace mlir::iree_compiler {

template <typename Ty>
struct opt_initializer {
  llvm::StringRef parentName;
  Ty init;
  llvm::OptimizationLevel optLevel;

  opt_initializer(llvm::StringRef parentName, const llvm::OptimizationLevel opt,
                  const Ty &val)
      : parentName(parentName), init(val), optLevel(opt) {}
  void apply(const llvm::OptimizationLevel inLevel, Ty &val) const {
    assert(inLevel.getSizeLevel() == 0 && "size level not implemented");
    if (inLevel.getSpeedupLevel() >= optLevel.getSpeedupLevel())
      val = init;
  }

  /// Append to the description string of the flag.
  /// e.g. " at O2 default is true"
  void appendToDesc(std::string &desc) {
    llvm::raw_string_ostream os(desc);
    os << "\nAt optimization level " << optLevel << " the default is ";
    prettyPrint(os, init);
  }

private:
  // TODO: merge this with the printing in `OptionsBinder`.
  template <typename T>
  static void prettyPrint(llvm::raw_ostream &os, T &val) {
    os << val;
  }

  void prettyPrint(llvm::raw_ostream &os, bool &val) {
    os << (val ? "true" : "false");
  }
};

struct opt_scope {
  llvm::StringRef name;

  template <typename T>
  opt_initializer<T> operator()(llvm::OptimizationLevel optLevel, T init) {
    return opt_initializer<T>(name, optLevel, init);
  }
};

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
    auto [changedCallback, clCallback] = makeChangedCallback<V>();
    OptionInfo &info = getOptionsStorage()[name];
    if (!scope) {
      // Bind global options.
      auto opt = std::make_unique<llvm::cl::opt<T, /*ExternalStorage=*/true>>(
          name, llvm::cl::location(value), llvm::cl::init(value), clCallback,
          std::forward<Mods>(Ms)...);
      auto defaultCallback = makeDefaultCallback(&value);
      info.option = std::move(opt);
      info.isChanged = changedCallback;
      info.isDefault = defaultCallback;
    } else {
      // Bind local options.
      auto option =
          std::make_unique<llvm::cl::opt<T, /*ExternalStorage=*/true>>(
              name, llvm::cl::sub(*scope), llvm::cl::location(value),
              llvm::cl::init(value), clCallback, std::forward<Mods>(Ms)...);
      auto printCallback =
          makePrintCallback(option->ArgStr, option->getParser(), &value);
      auto defaultCallback = makeDefaultCallback(&value);
      info.option = std::move(option);
      info.print = printCallback;
      info.isChanged = changedCallback;
      info.isDefault = defaultCallback;
    }
  }

  // Bind a flag with a single `opt_initialier` that specifies defaults at a
  // given optimization level.
  template <typename T, typename V, typename... Mods>
  void opt(llvm::StringRef name, V &value,
           std::initializer_list<opt_initializer<T>> inits, Mods... Ms) {
    llvm::SmallVector<opt_initializer<T>> initsSorted(inits.begin(),
                                                      inits.end());
    llvm::sort(initsSorted, [](opt_initializer<T> &lhs,
                               opt_initializer<T> &rhs) {
      return lhs.optLevel.getSpeedupLevel() < rhs.optLevel.getSpeedupLevel();
    });

    llvm::cl::desc &desc = filterDescription(Ms...);
    auto descStr = std::make_unique<std::string>(desc.Desc);
    for (auto &init : initsSorted) {
      init.appendToDesc(*descStr);
    }
    desc.Desc = descStr->c_str();

    opt<V>(name, value, Ms...);
    OptionInfo &info = getOptionsStorage()[name];
    info.extendedDesc = std::move(descStr);

    auto sharedValue = std::make_shared<V>();
    llvm::StringRef parentName = inits.begin()->parentName;

    auto changedCallback = info.isChanged;

    assert(getOptionsStorage().count(parentName) && "Parent option not found");
    auto &parentInfo = getOptionsStorage()[parentName];
    auto optDef = std::make_unique<OptOverride<T>>(
        value, std::move(initsSorted), info.isChanged);
    parentInfo.children->push_back(optDef.get());
    info.optOverrides = std::move(optDef);
  }

  template <typename... Mods>
  void topLevelOpt(llvm::StringRef name, llvm::OptimizationLevel &value,
                   Mods... Ms) {
    assert(getRootFlag() == kDummyRootFlag);
    auto &optionInfos = getOptionsStorage();
    auto dummyIt = optionInfos.find(kDummyRootFlag);
    if (dummyIt != optionInfos.end()) {
      optionInfos[name] = std::move(dummyIt->second);
      optionInfos.erase(dummyIt);
    }
    getRootFlag() = name;

    opt<llvm::OptimizationLevel>(name, value, Ms...);
    auto &info = optionInfos[name];
    info.optOverrides = std::make_unique<RootOptOverride>(value, info.isChanged,
                                                          *info.children);
  }

  void applyOptimizationDefaults() {
    // Recursively applies overrides.
    auto it = getOptionsStorage().find(getRootFlag());
    assert(it != getOptionsStorage().end() && "Option not found");
    it->second.optOverrides->applyFromRoot();
  }

  void restoreOptimizationDefaults() {
    for (auto &[_, info] : getOptionsStorage()) {
      if (info.optOverrides)
        info.optOverrides->restoreBackup();
    }
  }

  template <typename... Mods>
  opt_scope optimizationLevel(llvm::StringRef name,
                              llvm::OptimizationLevel &value, Mods... Ms) {
    opt<llvm::OptimizationLevel>(name, value, Ms...);
    auto &info = getOptionsStorage()[name];
    auto optDef = std::make_unique<ScopedOptOverride>(value, info.isChanged,
                                                      *info.children);
    getOptionsStorage()[getRootFlag()].children->push_back(optDef.get());
    info.optOverrides = std::move(optDef);
    return opt_scope{name};
  }

  template <typename T, typename V, typename... Mods>
  void list(llvm::StringRef name, V &value, Mods... Ms) {
    OptionInfo &info = getOptionsStorage()[name];
    if (!scope) {
      // Bind global options.
      auto list =
          std::make_unique<llvm::cl::list<T>>(name, std::forward<Mods>(Ms)...);
      // Since list does not support external storage, hook the callback
      // and use it to update.
      list->setCallback(
          [&value](const T &newElement) { value.push_back(newElement); });
      auto defaultCallback = makeListDefaultCallback(&value);
      info.option = std::move(list);
      info.isDefault = defaultCallback;
    } else {
      // Bind local options.
      auto list = std::make_unique<llvm::cl::list<T>>(
          name, llvm::cl::sub(*scope), std::forward<Mods>(Ms)...);
      auto printCallback =
          makeListPrintCallback(list->ArgStr, list->getParser(), &value);
      auto defaultCallback = makeListDefaultCallback(&value);
      // Since list does not support external storage, hook the callback
      // and use it to update.
      list->setCallback(
          [&value](const T &newElement) { value.push_back(newElement); });
      info.option = std::move(list);
      info.print = printCallback;
      info.isDefault = defaultCallback;
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
  class OptOverrideBase {
  public:
    virtual ~OptOverrideBase() = default;
    virtual void backupAndApplyOpt(llvm::OptimizationLevel opt) = 0;
    virtual void restoreBackup() = 0;
    virtual void applyFromRoot() { llvm_unreachable("Not implemented"); }
  };

  template <typename T>
  class OptOverride : public OptOverrideBase {
  public:
    OptOverride(T &value, llvm::SmallVector<opt_initializer<T>> inits,
                std::function<bool()> isChanged)
        : backup(value), value(&value), inits(std::move(inits)),
          isChanged(std::move(isChanged)) {}

    void backupAndApplyOpt(llvm::OptimizationLevel opt) override {
      backup = *value;
      if (!isChanged()) {
        for (auto &init : inits) {
          init.apply(opt, *value);
        }
      }
    }
    void restoreBackup() override { *value = backup; }

  private:
    T backup;
    T *value;
    llvm::SmallVector<opt_initializer<T>> inits;
    std::function<bool()> isChanged;
  };

  class ScopedOptOverride : public OptOverrideBase {
  public:
    ScopedOptOverride(llvm::OptimizationLevel &value,
                      std::function<bool()> isChanged,
                      llvm::SmallVector<OptOverrideBase *> &children)
        : backup(value), value(&value), isChanged(std::move(isChanged)),
          children(children) {}
    void backupAndApplyOpt(llvm::OptimizationLevel opt) override {
      backup = *value;
      if (!isChanged()) {
        *value = opt;
      }
      for (auto &child : children) {
        child->backupAndApplyOpt(*value);
      }
    }
    void restoreBackup() override { *value = backup; }

  protected:
    llvm::OptimizationLevel backup;
    llvm::OptimizationLevel *value;
    std::function<bool()> isChanged;
    llvm::SmallVector<OptOverrideBase *> &children;
  };

  class RootOptOverride : public ScopedOptOverride {
  public:
    using ScopedOptOverride::ScopedOptOverride;
    void applyFromRoot() override {
      for (auto &child : children) {
        child->backupAndApplyOpt(*value);
      }
    }
    void backupAndApplyOpt(llvm::OptimizationLevel opt) override {
      llvm_unreachable("Not implemented");
    }
    void restoreBackup() override {}
  };

  struct OptionInfo {
    OptionInfo()
        : children(std::make_unique<llvm::SmallVector<OptOverrideBase *>>()) {}

    using PrintCallback = std::function<void(llvm::raw_ostream &)>;
    using ChangedCallback = std::function<bool()>;
    using DefaultCallback = std::function<bool()>;
    std::unique_ptr<llvm::cl::Option> option = nullptr;
    PrintCallback print = nullptr;
    ChangedCallback isChanged = nullptr;
    DefaultCallback isDefault = nullptr;

    // For options with optimization level defaults.
    std::unique_ptr<OptOverrideBase> optOverrides = nullptr;
    std::unique_ptr<std::string> extendedDesc = nullptr;
    std::unique_ptr<llvm::SmallVector<OptOverrideBase *>> children = nullptr;
  };
  using OptionsStorage = llvm::DenseMap<llvm::StringRef, OptionInfo>;

  OptionsStorage &getOptionsStorage();
  const OptionsStorage &getOptionsStorage() const;
  llvm::StringRef &getRootFlag();

  OptionsBinder() = default;
  OptionsBinder(std::unique_ptr<llvm::cl::SubCommand> scope)
      : scope(std::move(scope)) {}

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
                  static_cast<int>(*value), OptionInfo::PrintCallback()) {
    return [optionName, &parser, value](llvm::raw_ostream &os) {
      llvm::StringRef valueName("<unknown>");
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
                  OptionInfo::PrintCallback()) {
    return [optionName, value](llvm::raw_ostream &os) {
      os << "--" << optionName << "=" << *value;
    };
  }

  // Bool scalar print specialization.
  template <typename ParserTy>
  static auto makePrintCallback(llvm::StringRef optionName, ParserTy &parser,
                                bool *value)
      -> decltype(static_cast<llvm::cl::basic_parser<bool> &>(parser),
                  OptionInfo::PrintCallback()) {
    return [optionName, value](llvm::raw_ostream &os) {
      os << "--" << optionName << "=";
      if (*value) {
        os << "true";
      } else {
        os << "false";
      }
    };
  }

  // Returns a pair of callbacks, the first returns if the option has been
  // parsed and the second is passed to llvm::cl to track if the option has been
  // parsed.
  template <typename V>
  static std::pair<OptionInfo::ChangedCallback, llvm::cl::cb<void, V>>
  makeChangedCallback() {
    std::shared_ptr<bool> changed = std::make_shared<bool>(false);
    return std::pair{
        [changed]() -> bool { return *changed; },
        llvm::cl::cb<void, V>([changed](const V &) { *changed = true; })};
  }

  // Scalar default specialization.
  template <typename V>
  static OptionInfo::DefaultCallback makeDefaultCallback(V *currentValue) {
    // Capture the current value as the initial value.
    V initialValue = *currentValue;
    return [currentValue, initialValue]() -> bool {
      return *currentValue != initialValue;
    };
  }

  // List default specialization.
  template <typename V>
  static OptionInfo::DefaultCallback makeListDefaultCallback(V *currentValue) {
    return [currentValue]() -> bool { return !currentValue->empty(); };
  }

  // List basic print specialization. This is the only one we provide so far.
  // Add others if the compiler tells you to.
  template <typename ListTy, typename ParserTy,
            typename V = typename ListTy::value_type>
  static auto makeListPrintCallback(llvm::StringRef optionName,
                                    ParserTy &parser, ListTy *values)
      -> decltype(static_cast<llvm::cl::basic_parser<V> &>(parser),
                  OptionInfo::PrintCallback()) {
    return [optionName, values](llvm::raw_ostream &os) {
      os << "--" << optionName << "=";
      for (auto it : llvm::enumerate(*values)) {
        if (it.index() > 0)
          os << ",";
        os << it.value();
      }
    };
  }

  // Finds the description in args
  template <typename... Args>
  static llvm::cl::desc &filterDescription(Args &...args) {
    llvm::cl::desc *result = nullptr;
    (
        [&] {
          if constexpr (std::is_same_v<std::decay_t<Args>, llvm::cl::desc>) {
            assert(!result && "Multiple llvm::cl::desc in args");
            if (!result)
              result = &args;
          }
        }(),
        ...);
    assert(result && "Expected llvm::cl::desc in args");
    return *result;
  }

  std::unique_ptr<llvm::cl::SubCommand> scope;

  OptionsStorage localOptions;
  static llvm::ManagedStatic<OptionsStorage> globalOptions;

  llvm::StringRef localRootFlag = kDummyRootFlag;
  static llvm::StringRef globalRootFlag;
  static constexpr llvm::StringRef kDummyRootFlag = "iree-dummy-opt-level-flag";
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

namespace llvm::cl {

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

template <>
class parser<OptimizationLevel> : public basic_parser<OptimizationLevel> {
public:
  parser(Option &O) : basic_parser(O) {}
  bool parse(Option &O, StringRef ArgName, StringRef Arg,
             OptimizationLevel &Val);
  StringRef getValueName() const override { return "optimization level"; }
  void printOptionDiff(const Option &O, OptimizationLevel V,
                       const OptVal &Default, size_t GlobalWidth) const;
  void anchor() override;
};

} // namespace llvm::cl

#endif // IREE_COMPILER_UTILS_FLAG_UTILS_H
