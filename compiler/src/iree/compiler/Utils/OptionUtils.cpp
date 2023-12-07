// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/OptionUtils.h"

#include "llvm/Support/ManagedStatic.h"

namespace mlir::iree_compiler {

void OptionsBinder::addGlobalOption(std::unique_ptr<llvm::cl::Option> option) {
  static llvm::ManagedStatic<std::vector<std::unique_ptr<llvm::cl::Option>>>
      globalOptions;
  globalOptions->push_back(std::move(option));
}

LogicalResult OptionsBinder::parseArguments(int argc, const char *const *argv,
                                            ErrorCallback onError) {
  assert(scope && "can only parse arguments for local scoped binder");
  for (int i = 0; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    llvm::StringRef nameVal;
    if (arg.startswith("--")) {
      nameVal = arg.drop_front(2);
    } else if (arg.startswith("-")) {
      nameVal = arg.drop_front(1);
    } else {
      // Pure positional options not supported.
      if (onError) {
        onError("pure positional arguments not supported (prefix with '--')");
      }
      return failure();
    }

    // Split name and value.
    llvm::StringRef name;
    llvm::StringRef value;
    size_t eqPos = nameVal.find("=");
    if (eqPos == llvm::StringRef::npos) {
      name = nameVal;
    } else {
      name = nameVal.take_front(eqPos);
      value = nameVal.drop_front(eqPos + 1);
    }

    // Find the option.
    auto foundIt = scope->OptionsMap.find(name);
    if (foundIt == scope->OptionsMap.end()) {
      if (onError) {
        std::string message("option not found: ");
        message.append(name.begin(), name.end());
        onError(message);
      }
      return failure();
    }
    llvm::cl::Option *option = foundIt->second;

    if (llvm::cl::ProvidePositionalOption(option, value, argc)) {
      // Error.
      if (onError) {
        std::string message("option parse error for: ");
        message.append(name.begin(), name.end());
        message.append("=");
        message.append(value.begin(), value.end());
        onError(message);
      }
      return failure();
    }
  }

  return success();
}

llvm::SmallVector<std::string>
OptionsBinder::printArguments(bool nonDefaultOnly) {
  llvm::SmallVector<std::string> values;
  for (auto &info : localOptions) {
    if (!info.print)
      continue;
    if (nonDefaultOnly && !info.isChanged())
      continue;

    std::string s;
    llvm::raw_string_ostream os(s);
    info.print(os);
    os.flush();
    values.push_back(std::move(s));
  }
  return values;
}

} // namespace mlir::iree_compiler
//
// Examples:
//   1073741824 => 1073741824
//          1gb => 1000000000
//         1gib => 1073741824
static int64_t ParseByteSize(llvm::StringRef value) {
  // TODO(benvanik): probably worth to-lowering here on the size. Having copies
  // of all the string view utils for just this case is code size overkill. For
  // now only accept lazy lowercase.
  int64_t scale = 1;
  if (value.consume_back_insensitive("kb")) {
    scale = 1000;
  } else if (value.consume_back_insensitive("kib")) {
    scale = 1024;
  } else if (value.consume_back_insensitive("mb")) {
    scale = 1000 * 1000;
  } else if (value.consume_back_insensitive("mib")) {
    scale = 1024 * 1024;
  } else if (value.consume_back_insensitive("gb")) {
    scale = 1000 * 1000 * 1000;
  } else if (value.consume_back_insensitive("gib")) {
    scale = 1024 * 1024 * 1024;
  } else if (value.consume_back_insensitive("b")) {
    scale = 1;
  }
  auto terminatedStr = value.str();
  int64_t size = std::atoll(terminatedStr.data());
  return size * scale;
}

namespace llvm {
namespace cl {
template class basic_parser<ByteSize>;
template class basic_parser<PowerOf2ByteSize>;
} // namespace cl
} // namespace llvm

using ByteSize = llvm::cl::ByteSize;
using PowerOf2ByteSize = llvm::cl::PowerOf2ByteSize;

// Return true on error.
bool llvm::cl::parser<ByteSize>::parse(Option &O, StringRef ArgName,
                                       StringRef Arg, ByteSize &Val) {
  Val.value = ParseByteSize(Arg);
  return false;
}

void llvm::cl::parser<ByteSize>::printOptionDiff(const Option &O, ByteSize V,
                                                 const OptVal &Default,
                                                 size_t GlobalWidth) const {
  printOptionName(O, GlobalWidth);
  std::string Str;
  {
    llvm::raw_string_ostream SS(Str);
    SS << V.value;
  }
  outs() << "= " << Str;
  outs().indent(2) << " (default: ";
  if (Default.hasValue()) {
    outs() << Default.getValue().value;
  } else {
    outs() << "*no default*";
  }
  outs() << ")\n";
}

void llvm::cl::parser<ByteSize>::anchor() {}

// Return true on error.
bool llvm::cl::parser<PowerOf2ByteSize>::parse(Option &O, StringRef ArgName,
                                               StringRef Arg,
                                               PowerOf2ByteSize &Val) {
  Val.value = ParseByteSize(Arg);
  if (!llvm::isPowerOf2_64(Val.value)) {
    return O.error("'" + Arg +
                   "' value not a power-of-two, use 16mib/64mib/2gb/etc");
    return true;
  }
  return false;
}

void llvm::cl::parser<PowerOf2ByteSize>::printOptionDiff(
    const Option &O, PowerOf2ByteSize V, const OptVal &Default,
    size_t GlobalWidth) const {
  printOptionName(O, GlobalWidth);
  std::string Str;
  {
    llvm::raw_string_ostream SS(Str);
    SS << V.value;
  }
  outs() << "= " << Str;
  outs().indent(2) << " (default: ";
  if (Default.hasValue()) {
    outs() << Default.getValue().value;
  } else {
    outs() << "*no default*";
  }
  outs() << ")\n";
}

void llvm::cl::parser<PowerOf2ByteSize>::anchor() {}
