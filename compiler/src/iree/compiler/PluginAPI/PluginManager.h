// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>
#include <string_view>
#include <vector>

#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::iree_compiler {

class PluginManager;
class PluginManagerSession;

// Command line options for the plugin manager.
class PluginManagerOptions {
 public:
  // Plugins to be activated in a session.
  llvm::SmallVector<std::string> plugins;

  // Print plugin information to stderr.
  bool printPluginInfo = false;

  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<PluginManagerOptions>;
};

// Manages global registrations for available plugins.
// Typically, there will be one PluginManager globally for the compiler, and
// it is initialized in ireeCompilerGlobalInitialize() based on statically
// compiled plugins or by querying environment variables and/or command
// line options to load dynamic plugins.
//
// At PluginManager initialization time, the only thing that is done is to
// record which plugins are registered and invoke their registration callback.
// This is responsible for registering flags and other global customizations.
//
// Most of the work of a plugin is done at session initialization time when
// an MLIRContext is available.
class PluginManager : public PluginRegistrar {
 public:
  PluginManager();

  // Initializes the plugin manager. Since this may do shared library opening
  // and use failable initializers, it can fail. There probably isn't much to
  // do in that case but crash, but the choice is left to the caller.
  bool loadAvailablePlugins();

  // Calls through to AbstractPluginRegistration::globalInitialize for all
  // available plugins.
  void globalInitialize();

  // Calls through to AbstractPluginRegistration::registerPasses for all
  // available plugins.
  void registerPasses();

  // Calls through to AbstractPluginRegistration::initializeCLI for all
  // available plugins.
  void initializeCLI();

  // Calls through to AbstractPluginRegistration::registerDialects for all
  // available plugins.
  void registerDialects(DialectRegistry &registry);

 private:
  friend class PluginManagerSession;
};

// Holds activated plugins for an |iree_compiler_session_t|.
class PluginManagerSession {
 public:
  PluginManagerSession(PluginManager &pluginManager, OptionsBinder &binder,
                       PluginManagerOptions &options);

  // Activates plugins as configured.
  LogicalResult activatePlugins(MLIRContext *context);

 private:
  PluginManagerOptions &options;
  // At construction, uninitialized plugin sessions are created for all
  // registered plugins so that CLI options can be set properly.
  llvm::StringMap<std::unique_ptr<AbstractPluginSession>> allPluginSessions;

  // Activation state.
  llvm::SmallVector<AbstractPluginSession *> activatedSessions;
};

}  // namespace mlir::iree_compiler
