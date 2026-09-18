// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/PluginManager.h"

#include <cstring>
#include <string>
#include <utility>

#include "iree/compiler/PluginAPI/PluginEntryPoint.h"

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

// Declare entrypoints for each statically registered plugin.
#define HANDLE_PLUGIN_ID(plugin_id)                                            \
  extern "C" bool iree_register_compiler_plugin_##plugin_id(                   \
      mlir::iree_compiler::PluginRegistrar *);
#include "iree/compiler/PluginAPI/Config/StaticLinkedPlugins.inc"
#undef HANDLE_PLUGIN_ID

IREE_DEFINE_COMPILER_OPTION_FLAGS(mlir::iree_compiler::PluginManagerOptions);

namespace mlir::iree_compiler {

void PluginManagerOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category("IREE compiler plugin options");

  binder.list<std::string>("iree-plugin", plugins,
                           llvm::cl::desc("Plugins to activate"),
                           llvm::cl::cat(category));
  binder.opt<bool>(
      "iree-print-plugin-info", printPluginInfo,
      llvm::cl::desc("Prints available and activated plugin info to stderr"),
      llvm::cl::cat(category));
}

namespace {

// --iree-load-plugin is parsed by hand before llvm::cl exists, so plugins can
// add options. This sink keeps cl from rejecting it and lists it in --help.
struct PluginOptionsSink {
  llvm::SmallVector<std::string> pluginOpts;
  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("IREE dynamic plugin options");
    binder.list<std::string>(
        "iree-load-plugin", pluginOpts,
        llvm::cl::desc("Shared library to load. The plugin reports its id; "
                       "activate it with --iree-plugin."),
        llvm::cl::cat(category));
  }
  using FromFlags = OptionsFromFlags<PluginOptionsSink>;
};

} // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(PluginOptionsSink);

DynamicPluginRegistry &DynamicPluginRegistry::get() {
  static DynamicPluginRegistry instance;
  return instance;
}

llvm::Error DynamicPluginRegistry::initialize(llvm::ArrayRef<const char *> args,
                                              EnvPlugins envPlugins) {
  assert(!initialized && "flags cannot be processed twice");
  initialized = true;

  // Registered even without a command line, so cl accepts and documents the
  // flag.
  (void)PluginOptionsSink::FromFlags::get();

  loadPluginsFromCL(args);
  if (envPlugins == EnvPlugins::Enabled) {
    loadPluginPathsFromEnv();
  }
  return std::move(loadErrors);
}

void DynamicPluginRegistry::addPlugin(llvm::Expected<Plugin> plugin) {
  if (!plugin) {
    loadErrors = llvm::joinErrors(std::move(loadErrors), plugin.takeError());
    return;
  }
  for (const Plugin &existing : plugins) {
    // Naming one library twice is ordinary.
    if (existing.path == plugin->path) {
      return;
    }
    // A second registration of an id aborts inside the registrar.
    if (existing.pluginId == plugin->pluginId) {
      loadErrors = llvm::joinErrors(
          std::move(loadErrors),
          llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "plugin '%s' reports id '%s', already taken by '%s'",
              plugin->path.c_str(), plugin->pluginId.c_str(),
              existing.path.c_str()));
      return;
    }
  }
  plugins.push_back(std::move(*plugin));
}

bool initializeDynamicPlugins(llvm::ArrayRef<const char *> args,
                              llvm::raw_ostream &os) {
  // The first caller's arguments win; later callers have none to add.
  static const bool valid = [&] {
    llvm::Error e = DynamicPluginRegistry::get().initialize(
        args, DynamicPluginRegistry::EnvPlugins::Enabled);
    if (!e) {
      return true;
    }
    // logAllUnhandledErrors prefixes only the first error.
    llvm::handleAllErrors(std::move(e), [&](const llvm::ErrorInfoBase &info) {
      os << "[IREE Dynamic Plugin ERROR]: " << info.message() << "\n";
    });
    return false;
  }();
  return valid;
}

bool DynamicPluginRegistry::registerPlugins(PluginRegistrar *registrar) const {
  bool success = true;
  for (const auto &plugin : plugins) {
    if (!plugin.registerFunction(registrar)) {
      llvm::errs() << "[IREE Dynamic Plugin ERROR]: registration function of '"
                   << plugin.pluginId << "' (" << plugin.path << ") failed\n";
      success = false;
    }
  }
  return success;
}

llvm::SmallVector<std::string> DynamicPluginRegistry::getLoadedPlugins() const {
  llvm::SmallVector<std::string> pluginNames;
  for (const auto &plugin : plugins) {
    pluginNames.push_back(plugin.pluginId);
  }
  return pluginNames;
}

bool DynamicPluginRegistry::hasPluginPath(llvm::StringRef path) const {
  return llvm::any_of(
      plugins, [&](const Plugin &plugin) { return plugin.path == path; });
}

bool verifyDynamicPluginFlags(llvm::raw_ostream &os) {
  const auto &registry = DynamicPluginRegistry::get();
  bool valid = true;
  for (const std::string &path :
       PluginOptionsSink::FromFlags::get().pluginOpts) {
    if (!registry.hasPluginPath(path)) {
      os << "[IREE Dynamic Plugin ERROR]: --iree-load-plugin=" << path
         << " was not loaded; this entry point takes plugins from "
            "IREE_LOAD_PLUGINS\n";
      valid = false;
    }
  }
  return valid;
}

llvm::Expected<DynamicPluginRegistry::Plugin>
DynamicPluginRegistry::Plugin::loadFromPath(llvm::StringRef path) {
  Plugin plugin;
  plugin.path = path.str();

#if defined(_WIN32)
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "dynamic plugins are not supported on Windows");
#else
  // llvm::sys::DynamicLibrary uses RTLD_LAZY, which defers a missing symbol to
  // its first call. RTLD_LOCAL keeps two plugins carrying the same helper from
  // binding to each other's copy.
  plugin.library = ::dlopen(plugin.path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!plugin.library) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "could not load plugin library '%s': %s",
                                   plugin.path.c_str(), ::dlerror());
  }

  // POSIX guarantees the cast.
  auto getInfo = reinterpret_cast<IreeCompilerPluginInfoGetter>(
      ::dlsym(plugin.library, IREE_COMPILER_PLUGIN_INFO_SYMBOL_NAME));
  if (!getInfo) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "plugin '%s' defines no %s; declare it with "
                                   "IREE_DEFINE_COMPILER_PLUGIN",
                                   plugin.path.c_str(),
                                   IREE_COMPILER_PLUGIN_INFO_SYMBOL_NAME);
  }

  const IreeCompilerPluginInfo *info = getInfo();
  if (!info) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "plugin '%s' returned no plugin info",
                                   plugin.path.c_str());
  }
  if (info->apiVersion != IREE_COMPILER_PLUGIN_API_VERSION) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "plugin '%s' was built against plugin API version %u, this compiler "
        "speaks %u",
        plugin.path.c_str(), info->apiVersion,
        IREE_COMPILER_PLUGIN_API_VERSION);
  }

  // These fields exist only in the version checked above.
  if (!info->abiHash || !info->pluginId || !info->pluginVersion ||
      !info->registerPlugin) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "plugin '%s' left a field of IreeCompilerPluginInfo null",
        plugin.path.c_str());
  }
  if (std::strcmp(info->abiHash, IREE_COMPILER_PLUGIN_ABI_HASH) != 0) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "plugin '%s' (%s) was built against plugin API headers %s, this "
        "compiler has %s",
        plugin.path.c_str(), info->pluginVersion, info->abiHash,
        IREE_COMPILER_PLUGIN_ABI_HASH);
  }

  plugin.pluginId = info->pluginId;
  plugin.registerFunction = info->registerPlugin;
  return plugin;
#endif // _WIN32
}

void DynamicPluginRegistry::loadPluginsFromCL(
    llvm::ArrayRef<const char *> args) {
  // Match every spelling cl accepts: one or two dashes, a separated value,
  // response files. Anything missed here the sink swallows and nothing loads.
  llvm::BumpPtrAllocator alloc;
  llvm::SmallVector<const char *> expanded(args.begin(), args.end());
  llvm::cl::ExpansionContext expansion(alloc, llvm::cl::TokenizeGNUCommandLine);
  if (llvm::Error e = expansion.expandResponseFiles(expanded)) {
    loadErrors = llvm::joinErrors(std::move(loadErrors), std::move(e));
  }

  for (size_t i = 1; i < expanded.size(); ++i) {
    llvm::StringRef arg = expanded[i];
    // cl treats everything after -- as positional.
    if (arg == "--") {
      break;
    }
    if (!arg.consume_front("--") && !arg.consume_front("-")) {
      continue;
    }
    if (!arg.consume_front("iree-load-plugin")) {
      continue;
    }
    llvm::StringRef path;
    if (arg.consume_front("=")) {
      path = arg;
    } else if (arg.empty() && i + 1 < expanded.size()) {
      path = expanded[++i];
    } else {
      // A longer flag with the same prefix.
      continue;
    }
    if (path.empty()) {
      loadErrors = llvm::joinErrors(
          std::move(loadErrors),
          llvm::createStringError(llvm::inconvertibleErrorCode(),
                                  "no path given to --iree-load-plugin"));
      continue;
    }
    addPlugin(Plugin::loadFromPath(path));
  }
}

void DynamicPluginRegistry::loadPluginPathsFromEnv() {
  std::optional<std::string> envVar =
      llvm::sys::Process::GetEnv("IREE_LOAD_PLUGINS");
  if (!envVar) {
    return;
  }
  llvm::SmallVector<llvm::StringRef, 4> paths;
  llvm::StringRef(*envVar).split(paths, ',', -1, /*KeepEmpty=*/false);
  for (llvm::StringRef path : paths) {
    addPlugin(Plugin::loadFromPath(path));
  }
}

bool PluginManager::loadAvailablePlugins() {
// Initialize static plugins.
#define HANDLE_PLUGIN_ID(plugin_id)                                            \
  if (!iree_register_compiler_plugin_##plugin_id(this))                        \
    return false;
#include "iree/compiler/PluginAPI/Config/StaticLinkedPlugins.inc"
#undef HANDLE_PLUGIN_ID

  // The registrar aborts on a duplicate id, and dynamic ids are user input.
  auto &registry = DynamicPluginRegistry::get();
  bool unique = true;
  for (const std::string &pluginId : registry.getLoadedPlugins()) {
    if (registrations.count(pluginId)) {
      llvm::errs() << "[IREE Dynamic Plugin ERROR]: '" << pluginId
                   << "' has the same id as an already registered plugin\n";
      unique = false;
    }
  }
  if (!unique) {
    return false;
  }

  return registry.registerPlugins(this);
}

void PluginManager::globalInitialize() {
  for (auto &kv : registrations) {
    kv.second->globalInitialize();
  }
}

void PluginManager::registerPasses() {
  for (auto &kv : registrations) {
    kv.second->registerPasses();
  }
}

void PluginManager::initializeCLI() {
  for (auto &kv : registrations) {
    kv.second->initializeCLI();
  }
}

void PluginManager::registerGlobalDialects(DialectRegistry &registry) {
  for (auto &kv : registrations) {
    kv.second->registerGlobalDialects(registry);
  }
}

llvm::SmallVector<std::string> PluginManager::getLoadedPlugins() const {
  llvm::SmallVector<std::string> plugins;
#define HANDLE_PLUGIN_ID(plugin_id) plugins.push_back(#plugin_id);
#include "iree/compiler/PluginAPI/Config/StaticLinkedPlugins.inc"
#undef HANDLE_PLUGIN_ID

  auto dynamicPlugins = DynamicPluginRegistry::get().getLoadedPlugins();
  plugins.append(dynamicPlugins.begin(), dynamicPlugins.end());

  return plugins;
}

PluginManagerSession::PluginManagerSession(PluginManager &pluginManager,
                                           OptionsBinder &binder,
                                           PluginManagerOptions &options)
    : options(options) {
  for (auto &kv : pluginManager.registrations) {
    std::unique_ptr<AbstractPluginSession> session =
        kv.second->createUninitializedSession(binder);
    if (kv.second->getActivationPolicy() ==
        PluginActivationPolicy::DefaultActivated) {
      defaultActivatedSessions.insert(
          std::make_pair(kv.first(), session.get()));
    }
    allPluginSessions.insert(std::make_pair(kv.first(), std::move(session)));
  }
}

LogicalResult PluginManagerSession::initializePlugins() {
  auto getAvailableIds = [&]() -> llvm::SmallVector<llvm::StringRef> {
    llvm::SmallVector<llvm::StringRef> availableIds;
    for (auto &kv : allPluginSessions) {
      availableIds.push_back(kv.first());
    }
    std::sort(availableIds.begin(), availableIds.end());
    return availableIds;
  };

  // Print available plugins.
  if (options.printPluginInfo) {
    // Get the available plugins.
    llvm::errs() << "[IREE plugins]: Available plugins: ";
    llvm::interleaveComma(getAvailableIds(), llvm::errs());
    llvm::errs() << "\n";
  }

  // Loop through listed plugins and any that start with "-" go in the
  // set of disabled ids. This will be used to disable default activations.
  llvm::StringSet<> disabledIds;
  for (auto &pluginId : options.plugins) {
    if (llvm::StringRef(pluginId).starts_with("-")) {
      disabledIds.insert(llvm::StringRef(pluginId).substr(1));
    }
  }

  // Process default activated plugins.
  llvm::StringSet<> initializedIds;
  for (auto &it : defaultActivatedSessions) {
    if (disabledIds.contains(it.first())) {
      if (options.printPluginInfo) {
        llvm::errs() << "[IREE plugins]: Skipping disabled default '"
                     << it.first() << "'\n";
      }
      continue;
    }

    // Skip if already initialized.
    if (!initializedIds.insert(it.first()).second) {
      continue;
    }

    if (options.printPluginInfo) {
      llvm::errs() << "[IREE plugins]: Initializing default '" << it.first()
                   << "'\n";
    }
    initializedSessions.push_back(it.second);
  }

  // Process activations.
  // In the future, we may make this smarter by allowing dependencies and
  // sorting accordingly. For now, what you say is what you get.
  for (auto &pluginId : options.plugins) {
    if (llvm::StringRef(pluginId).starts_with("-")) {
      // Skip: It has already been added to disabledIds.
      continue;
    }

    // Skip if already initialized.
    if (!initializedIds.insert(pluginId).second) {
      continue;
    }

    if (options.printPluginInfo) {
      llvm::errs() << "[IREE plugins]: Initializing plugin '" << pluginId
                   << "'\n";
    }
    auto foundIt = allPluginSessions.find(pluginId);
    if (foundIt == allPluginSessions.end()) {
      llvm::errs() << "[IREE plugins error]: could not activate requested "
                      "IREE plugin '"
                   << pluginId
                   << "' because it is not registered (available plugins: ";
      llvm::interleaveComma(getAvailableIds(), llvm::errs());
      llvm::errs() << ")\n";
      return failure();
    }

    initializedSessions.push_back(foundIt->second.get());
  }

  return success();
}

void PluginManagerSession::registerDialects(DialectRegistry &registry) {
  for (auto *s : initializedSessions) {
    s->registerDialects(registry);
  }
}

LogicalResult PluginManagerSession::activatePlugins(MLIRContext *context) {
  for (auto *s : initializedSessions) {
    if (failed(s->activate(context))) {
      return failure();
    }
  }
  return success();
}

void PluginManagerSession::populateHALTargetDevices(
    IREE::HAL::TargetDeviceList &list) {
  for (auto *s : initializedSessions) {
    s->populateHALTargetDevices(list);
  }
}

void PluginManagerSession::populateHALTargetBackends(
    IREE::HAL::TargetBackendList &list) {
  for (auto *s : initializedSessions) {
    s->populateHALTargetBackends(list);
  }
}

} // namespace mlir::iree_compiler
