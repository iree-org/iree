// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/PluginManager.h"

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/DynamicLibrary.h"
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

std::optional<DynamicPluginRegistry> dynamicPluginRegistryInstance;

// Plugin options are parsed by hand, before llvm::cl exists, so that plugins
// can add their own. This swallows the flag afterwards so cl neither rejects it
// nor leaves it out of --help.
struct PluginOptionsSink {
  llvm::SmallVector<std::string> pluginOpts;
  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("IREE dynamic plugin options");
    binder.list<std::string>(
        "iree-load-plugin", pluginOpts,
        llvm::cl::desc(
            "Dynamic plugins to load, specified as <plugin_id>=<path> where "
            "path is the plugin shared library."),
        llvm::cl::cat(category));
  }
  using FromFlags = OptionsFromFlags<PluginOptionsSink>;
};

} // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(PluginOptionsSink);

bool DynamicPluginRegistry::create(int argc, char **argv,
                                   bool allowEnvPlugins) {
  if (dynamicPluginRegistryInstance.has_value()) {
    // The flags cannot be processed twice.
    return false;
  }

  dynamicPluginRegistryInstance = DynamicPluginRegistry();

  if (argv) {
    dynamicPluginRegistryInstance->loadPluginsFromCL(argc, argv);

    // Also consume the options to avoid unknown argument errors.
    (void)PluginOptionsSink::FromFlags::get();
  }

  if (allowEnvPlugins) {
    dynamicPluginRegistryInstance->loadPluginPathsFromEnv();
  }

  return dynamicPluginRegistryInstance->isValid();
}

bool DynamicPluginRegistry::hasInstance() {
  return dynamicPluginRegistryInstance.has_value();
}

DynamicPluginRegistry &DynamicPluginRegistry::get() {
  assert(dynamicPluginRegistryInstance.has_value() &&
         "DynamicPluginRegistry not initialized, call create() first");
  return dynamicPluginRegistryInstance.value();
}

bool DynamicPluginRegistry::registerPlugins(PluginRegistrar *registrar) const {
  bool success = true;
  for (const auto &plugin : plugins) {
    if (plugin.isValid()) {
      success = success && plugin.registerFunction(registrar);
    }
  }
  return success;
}

llvm::SmallVector<std::string> DynamicPluginRegistry::getLoadedPlugins() const {
  llvm::SmallVector<std::string> pluginNames;
  for (const auto &plugin : plugins) {
    if (plugin.isValid()) {
      pluginNames.push_back(plugin.pluginId);
    }
  }
  return pluginNames;
}

void DynamicPluginRegistry::reportErrors(llvm::raw_ostream &os) const {
  for (const auto &plugin : plugins) {
    if (!plugin.isValid()) {
      os << "[IREE Dynamic Plugin ERROR]: " << plugin.error.value() << "\n";
    }
  }
}

bool DynamicPluginRegistry::isValid() const {
  for (const auto &plugin : plugins) {
    if (!plugin.isValid()) {
      return false;
    }
  }
  return true;
}

DynamicPluginRegistry::Plugin
DynamicPluginRegistry::Plugin::loadFromString(std::string_view str) {
  Plugin plugin;

  auto equalPos = str.find('=');
  if (equalPos == std::string_view::npos) {
    plugin.error = "malformed plugin option '" + std::string(str) +
                   "', expected <plugin_id>=<path>";
    return plugin;
  }
  plugin.pluginId = std::string(str.substr(0, equalPos));
  plugin.path = std::string(str.substr(equalPos + 1));

  std::string loadErrMsg;
  plugin.library = llvm::sys::DynamicLibrary::getPermanentLibrary(
      plugin.path.c_str(), &loadErrMsg);
  if (!plugin.library.isValid()) {
    plugin.error =
        "could not load plugin library '" + plugin.path + "': " + loadErrMsg;
    return plugin;
  }

  std::string symbolName = "iree_register_compiler_plugin_" + plugin.pluginId;
  plugin.registerFunction = reinterpret_cast<RegisterFunction>(
      plugin.library.getAddressOfSymbol(symbolName.c_str()));
  if (!plugin.registerFunction) {
    plugin.error = "could not find registration function '" + symbolName +
                   "' in plugin '" + plugin.path + "'";
  }

  return plugin;
}

void DynamicPluginRegistry::loadPluginsFromCL(int argc, char **argv) {
  // Format: --iree-load-plugin=<plugin_id>=<path>

  for (int i = 1; i < argc; ++i) {
    StringRef argStr = argv[i];
    constexpr StringRef pluginOptionPrefix = "--iree-load-plugin";
    if (!argStr.starts_with(pluginOptionPrefix)) {
      continue;
    }
    if (argStr.size() <= pluginOptionPrefix.size() ||
        argStr[pluginOptionPrefix.size()] != '=') {
      Plugin plugin;
      plugin.error =
          "malformed iree-load-plugin option '" + std::string(argStr) +
          "', expected format '--iree-load-plugin=<plugin_id>=<path>'";
      plugins.push_back(std::move(plugin));
      continue;
    }
    StringRef pluginInfo = argStr.drop_front(pluginOptionPrefix.size() + 1);
    plugins.push_back(Plugin::loadFromString(pluginInfo));
  }
}

void DynamicPluginRegistry::loadPluginPathsFromEnv() {
  if (const char *envVar = std::getenv("IREE_LOAD_PLUGINS")) {
    StringRef envVarStr = envVar;
    SmallVector<StringRef, 4> pluginInfos;
    envVarStr.split(pluginInfos, ',', -1, false);
    for (StringRef pluginInfo : pluginInfos) {
      plugins.push_back(Plugin::loadFromString(pluginInfo));
    }
  }
}

PluginManager::PluginManager() = default;

bool PluginManager::loadAvailablePlugins() {
// Initialize static plugins.
#define HANDLE_PLUGIN_ID(plugin_id)                                            \
  if (!iree_register_compiler_plugin_##plugin_id(this))                        \
    return false;
#include "iree/compiler/PluginAPI/Config/StaticLinkedPlugins.inc"
#undef HANDLE_PLUGIN_ID

  // Initialize dynamic plugins.
  if (!DynamicPluginRegistry::get().registerPlugins(this)) {
    return false;
  }

  return true;
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

llvm::SmallVector<std::string> PluginManager::getLoadedPlugins() {
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
