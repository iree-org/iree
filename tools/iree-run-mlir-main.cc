// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE source.mlir/mlirbc -> execution output runner.
// This is meant to be called from LIT for FileCheck tests or as a developer
// tool to emulate what an online compiler does. It tries to match the interface
// of iree-compile so it's easy to run tests or approximate an
// `iree-compile | iree-run-module` sequence. If you want a more generalized
// runner for standalone precompiled IREE modules use iree-run-module instead.
//
// If there's a single exported function that will be executed and if there are
// multiple functions --function= can be used to specify which is executed.
// Function inputs can be provided with --input=. Results from the executed
// function will be printed to stdout for checking or can be written to files
// with --output=.
//
// Similar to iree-run-module the --device= flag can be used to specify which
// drivers and devices should be used to execute the function. The tool will
// try to infer which iree-compile flags are required for the devices used but
// this can be overridden by passing the --iree-hal-target-backends= and related
// flags explicitly. Likewise if only the target backend is specified the
// devices to use will be inferred unless explicitly specified.
//
// Example usage to compile and run with CUDA:
// $ iree-run-mlir --device=cuda://0 file.mlir
// or to compile with the LLVM CPU backend and run with the local-task driver:
// $ iree-run-mlir file.mlir \
//       --Xcompiler,iree-hal-target-backends=llvm-cpu --device=local-task
//
// Example usage in a lit test:
//   // RUN: iree-run-mlir --device= %s --function=foo --input=2xf32=2,3 | \
//   // RUN:   FileCheck %s
//   // CHECK-LABEL: @foo
//   // CHECK: 2xf32=[2 3]
//   func.func @foo(%arg0: tensor<2xf32>) -> tensor<2xf32> {
//     return %arg0 : tensor<2xf32>
//   }
//
// Command line arguments are handled by LLVM's parser by default but -- can be
// used to separate the compiler flags from the runtime flags, such as:
// $ iree-run-mlir source.mlir --device=local-task -- \
//       --iree-hal-target-backends=llvm-cpu
//
// In addition compiler/runtime flags can be passed in any order by prefixing
// them with --Xcompiler or --Xruntime like `--Xruntime,device=local-task` or
// `--Xruntime --device=local-task`.

#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/compiler/embedding_api.h"
#include "iree/hal/api.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/device_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/api.h"

namespace iree {
namespace {

// Polyfill for std::string_view::starts_with, coming in C++ 20.
// https://en.cppreference.com/w/cpp/string/basic_string_view/starts_with
bool starts_with(std::string_view prefix, std::string_view in_str) {
  return in_str.size() >= prefix.size() &&
         in_str.compare(0, prefix.size(), prefix) == 0;
}

// Tries to guess a default device name from the |target_backend| when possible.
// Users are still able to override this by passing in --device= flags.
std::string InferDefaultDeviceFromTargetBackend(
    std::string_view target_backend) {
  if (target_backend == "" || target_backend == "vmvx-inline") {
    // Plain VM or vmvx-inline targets do not need a HAL device.
    return "";
  } else if (target_backend == "llvm-cpu" || target_backend == "vmvx") {
    // Locally-executable targets default to the multithreaded task system
    // driver; users can override by specifying --device=local-sync instead.
    return "local-task";
  }
  // Many other backends use the `driver-pipeline` naming like `vulkan-spirv`
  // and we try that; device creation will fail if it's a bad guess.
  size_t dash = target_backend.find('-');
  if (dash == std::string::npos) {
    return std::string(target_backend);
  } else {
    return std::string(target_backend.substr(0, dash));
  }
}

// Tries to guess a target backend from the given |device_uri| when possible.
// Returns empty string if no backend is required or one could not be inferred.
std::string InferTargetBackendFromDevice(iree_string_view_t device_uri) {
  // Get the driver name from URIs in the `driver://...` form.
  iree_string_view_t driver = iree_string_view_empty();
  iree_string_view_split(device_uri, ':', &driver, nullptr);
  if (iree_string_view_is_empty(driver)) {
    // Plain VM or vmvx-inline targets do not need a HAL device.
    return "";
  } else if (iree_string_view_starts_with(driver, IREE_SV("local-"))) {
    // Locally-executable devices default to the llvm-cpu target as that's
    // usually what people want for CPU execution; users can override by
    // specifying --iree-hal-target-backends=vmvx instead.
    return "llvm-cpu";
  }
  // Many other backends have aliases that allow using the driver name. If there
  // are multiple pipelines available whatever the compiler defaults to is
  // chosen.
  return std::string(driver.data, driver.size);
}

// Tries to guess a set of target backends from the |device_flag_values| when
// possible. Since multiple target backends can be used for a particular device
// (such as llvm-cpu or vmvx for local-sync and local-task) this is just
// guesswork. If we can't produce a target backend flag value we bail.
// Returns a comma-delimited list of target backends.
StatusOr<std::string> InferTargetBackendsFromDevices(
    iree_string_view_list_t device_uris) {
  // No-op when no devices specified (probably no HAL).
  if (device_uris.count == 0) return "";
  // If multiple devices were provided we need to target all of them.
  std::set<std::string> target_backends;
  for (iree_host_size_t i = 0; i < device_uris.count; ++i) {
    auto target_backend = InferTargetBackendFromDevice(device_uris.values[i]);
    if (!target_backend.empty()) {
      target_backends.insert(std::move(target_backend));
    }
  }
  // Join all target backends together.
  std::string result;
  for (auto& target_backend : target_backends) {
    if (!result.empty()) result.append(",");
    result.append(target_backend);
  }
  return result;
}

// Configures the --iree-hal-target-backends= flag based on the --device= flags
// set by the user. Ignored if any target backends are explicitly specified.
// Online compilers would want to do some more intelligent device selection on
// their own.
Status ConfigureTargetBackends(iree_compiler_session_t* session,
                               std::string* out_default_device_uri) {
  // Query the session for the currently set --iree-hal-target-backends= flag.
  // It may be empty string.
  std::string target_backends_flag;
  ireeCompilerSessionGetFlags(
      session, /*nonDefaultOnly=*/true,
      [](const char* flag_str, size_t length, void* user_data) {
        // NOTE: flag_str has the full `--flag=value` string.
        std::string_view prefix = "--iree-hal-target-backends=";
        std::string_view flag = std::string_view(flag_str, length);
        if (starts_with(prefix, flag)) {
          flag.remove_prefix(prefix.size());
          if (flag.empty()) return;  // ignore empty
          auto* result = static_cast<std::string*>(user_data);
          *result = std::string(flag);
        }
      },
      static_cast<void*>(&target_backends_flag));

  // Query the tooling utils for the --device= flag values. Note that zero or
  // more devices may be specified.
  iree_string_view_list_t device_uris = iree_hal_device_flag_list();

  // No-op if no target backends or devices are specified - this can be an
  // intentional decision as the user may be running a program that doesn't use
  // the HAL.
  if (target_backends_flag.empty() && device_uris.count == 0) {
    return OkStatus();
  }

  // No-op if both target backends and devices are set as the user has
  // explicitly specified a configuration.
  if (!target_backends_flag.empty() && device_uris.count > 0) {
    return OkStatus();
  }

  // If target backends are specified then we can infer the runtime devices from
  // the compiler configuration. This only works if there's a single backend
  // specified; if the user wants multiple target backends then they must
  // specify the device(s) to use.
  if (device_uris.count == 0) {
    if (target_backends_flag.find(',') != std::string::npos) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "if multiple target backends are specified the device to use must "
          "also be specified with --device= (have "
          "`--iree-hal-target-backends=%.*s`)",
          (int)target_backends_flag.size(), target_backends_flag.data());
    }
    *out_default_device_uri =
        InferDefaultDeviceFromTargetBackend(target_backends_flag);
    return OkStatus();
  }

  // Infer target backends from the runtime device configuration.
  // This can get arbitrarily complex but for now this simple runner just
  // guesses. In the future we'll have more ways of configuring the compiler
  // from available runtime devices (not just the target backend but also
  // target-specific settings).
  IREE_ASSIGN_OR_RETURN(auto target_backends,
                        InferTargetBackendsFromDevices(device_uris));
  if (!target_backends.empty()) {
    auto target_backends_flag =
        std::string("--iree-hal-target-backends=") + target_backends;
    const char* compiler_argv[1] = {
        target_backends_flag.c_str(),
    };
    auto error = ireeCompilerSessionSetFlags(
        session, IREE_ARRAYSIZE(compiler_argv), compiler_argv);
    if (error) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unable to set inferred target backend flag to `%.*s`",
          (int)target_backends_flag.size(), target_backends_flag.data());
    }
  }

  return OkStatus();
}

// Runs the given .mlir file based on the current flags.
StatusOr<int> CompileAndRunFile(iree_compiler_session_t* session,
                                const char* mlir_filename) {
  IREE_TRACE_SCOPE_NAMED("CompileAndRunFile");

  // Configure the --iree-hal-target-backends= flag and/or get the default
  // device to use at runtime if either are not explicitly specified.
  // Note that target backends and the runtime devices aren't 1:1 and this is
  // an imperfect guess. In this simple online compiler we assume homogenous
  // device sets and only a single global target backend but library/hosting
  // layers can configure heterogenous and per-invocation target configurations.
  std::string default_device_uri;
  IREE_RETURN_IF_ERROR(ConfigureTargetBackends(session, &default_device_uri));

  // RAII container for the compiler invocation.
  struct InvocationState {
    iree_compiler_invocation_t* invocation = nullptr;
    iree_compiler_source_t* source = nullptr;
    iree_compiler_output_t* output = nullptr;

    explicit InvocationState(iree_compiler_session_t* session) {
      invocation = ireeCompilerInvocationCreate(session);
    }

    ~InvocationState() {
      if (source) ireeCompilerSourceDestroy(source);
      if (output) ireeCompilerOutputDestroy(output);
      ireeCompilerInvocationDestroy(invocation);
    }

    Status emitError(iree_compiler_error_t* error,
                     iree_status_code_t status_code,
                     std::string_view while_performing = "") {
      const char* msg = ireeCompilerErrorGetMessage(error);
      fprintf(stderr, "error compiling input file: %s\n", msg);
      iree_status_t status = iree_make_status(status_code, msg);
      if (!while_performing.empty()) {
        status = iree_status_annotate(
            status, iree_make_string_view(while_performing.data(),
                                          while_performing.size()));
      }
      ireeCompilerErrorDestroy(error);
      return status;
    }
  } state(session);

  // Open the source file on disk or stdin if `-`.
  if (auto error =
          ireeCompilerSourceOpenFile(session, mlir_filename, &state.source)) {
    return state.emitError(error, IREE_STATUS_NOT_FOUND, "opening source file");
  }

  // Open a writeable memory buffer that we can stream the compilation outputs
  // into. This may be backed by a memory-mapped file to allow for very large
  // results.
  if (auto error = ireeCompilerOutputOpenMembuffer(&state.output)) {
    return state.emitError(error, IREE_STATUS_INTERNAL,
                           "open output memory buffer");
  }

  // TODO: make parsing/pipeline execution return an error object.
  // We could capture diagnostics, stash them on the state, and emit with
  // ireeCompilerInvocationEnableCallbackDiagnostics.
  // For now we route all errors to stderr.
  ireeCompilerInvocationEnableConsoleDiagnostics(state.invocation);

  // Parse the source MLIR input and log verbose errors. Syntax errors or
  // version mismatches will hit here.
  if (!ireeCompilerInvocationParseSource(state.invocation, state.source)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to parse input file");
  }

  // Invoke the standard compilation pipeline to produce the compiled module.
  if (!ireeCompilerInvocationPipeline(state.invocation,
                                      IREE_COMPILER_PIPELINE_STD)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "failed to invoke main compiler pipeline");
  }

  // Flush the output to the memory buffer.
  if (auto error = ireeCompilerInvocationOutputVMBytecode(state.invocation,
                                                          state.output)) {
    return state.emitError(error, IREE_STATUS_INTERNAL,
                           "emitting output VM module binary");
  }

  // Get a raw host pointer to the output that we can pass to the runtime.
  void* binary_data = nullptr;
  uint64_t binary_size = 0;
  if (auto error = ireeCompilerOutputMapMemory(state.output, &binary_data,
                                               &binary_size)) {
    return state.emitError(error, IREE_STATUS_INTERNAL,
                           "mapping output buffer");
  }

  // Hosting libraries can route all runtime allocations to their own allocator
  // for statistics, isolation, or efficiency. Here we use the system
  // malloc/free.
  iree_allocator_t host_allocator = iree_allocator_system();

  // The same VM instance should be shared across many contexts. Here we only
  // use this once but a library would want to retain this and the devices it
  // creates for as long as practical.
  vm::ref<iree_vm_instance_t> instance;
  IREE_RETURN_IF_ERROR(iree_tooling_create_instance(host_allocator, &instance),
                       "creating instance");

  // Run the compiled module using the global flags for I/O (if any).
  // This loads the module, creates a VM context with it and any dependencies,
  // parses inputs from flags, and routes/verifies outputs as specified. Hosting
  // libraries should always reuse contexts if possible to amortize loading
  // costs and carry state (variables/etc) across invocations.
  //
  // This returns a process exit code based on the run mode (verifying expected
  // outputs, etc) that may be non-zero even if the status is success
  // ("execution completed successfully but values did not match").
  int exit_code = EXIT_SUCCESS;
  IREE_RETURN_IF_ERROR(
      iree_tooling_run_module_with_data(
          instance.get(),
          iree_make_string_view(default_device_uri.data(),
                                default_device_uri.size()),
          iree_make_const_byte_span(binary_data, (iree_host_size_t)binary_size),
          host_allocator, &exit_code),
      "running compiled module");
  return exit_code;
}

// Parses a combined list of compiler and runtime flags.
// Each argument list is stored in canonical argc/argv format with a trailing
// NULL string in the storage (excluded from the count).
class ArgParser {
 public:
  int compiler_argc() { return compiler_args_.size() - 1; }
  const char** compiler_argv() {
    return const_cast<const char**>(compiler_args_.data());
  }

  int runtime_argc() { return runtime_args_.size() - 1; }
  char** runtime_argv() { return runtime_args_.data(); }

  // Parses arguments from a raw command line argc/argv set.
  // Returns true if parsing was successful.
  bool Parse(int argc_raw, char** argv_raw) {
    // Pre-process the arguments with the compiler's argument parser since it
    // has super-powers on Windows and must work on the default main arguments.
    ireeCompilerGetProcessCLArgs(&argc_raw,
                                 const_cast<const char***>(&argv_raw));

    // Always add the progname to both flag sets.
    compiler_args_.push_back(argv_raw[0]);
    runtime_args_.push_back(argv_raw[0]);

    // Everything before -- goes to the runtime.
    // Everything after -- goes to the compiler.
    // To make it easier to form command lines in scripts we also allow
    // prefixing flags with -Xcompiler/-Xruntime on either side of the --.
    bool parsing_runtime_args = true;
    for (int i = 1; i < argc_raw; ++i) {
      char* current_arg_cstr = argv_raw[i];
      char* next_arg_cstr =
          argv_raw[i + 1];  // ok because list is NULL-terminated
      auto current_arg = std::string_view(current_arg_cstr);
      if (current_arg == "--") {
        // Switch default parsing to compiler flags.
        parsing_runtime_args = false;
      } else if (current_arg == "-Xcompiler" || current_arg == "--Xcompiler") {
        // Next arg is routed to the compiler.
        compiler_args_.push_back(next_arg_cstr);
      } else if (current_arg == "-Xruntime" || current_arg == "--Xruntime") {
        // Next arg is routed to the runtime.
        runtime_args_.push_back(next_arg_cstr);
      } else if (starts_with("-Xcompiler,", current_arg) ||
                 starts_with("--Xcompiler,", current_arg)) {
        // Split and send the rest of the flag to the compiler.
        AppendPrefixedArg(current_arg, &compiler_args_);
      } else if (starts_with("-Xruntime,", current_arg) ||
                 starts_with("--Xruntime,", current_arg)) {
        // Split and send the rest of the flag to the runtime.
        AppendPrefixedArg(current_arg, &runtime_args_);
      } else {
        // Route to either runtime or compiler arg sets based on which side of
        // the -- we are on.
        if (parsing_runtime_args) {
          runtime_args_.push_back(current_arg_cstr);
        } else {
          compiler_args_.push_back(current_arg_cstr);
        }
      }
    }

    // Add nullptrs to end to match real argv behavior.
    compiler_args_.push_back(nullptr);
    runtime_args_.push_back(nullptr);

    return true;
  }

 private:
  // Drops the prefix from |prefixed_arg| and appends the arg to |out_args|.
  // Example: --Xcompiler,ab=cd,ef -> --ab=cd,ef
  void AppendPrefixedArg(std::string_view prefixed_arg,
                         std::vector<char*>* out_args) {
    std::string_view sub_arg = prefixed_arg.substr(prefixed_arg.find(',') + 1);
    auto stable_arg = std::make_unique<std::string>("--");
    stable_arg->append(sub_arg);
    temp_strings_.push_back(std::move(stable_arg));
    out_args->push_back(temp_strings_.back()->data());
  }

  std::vector<std::unique_ptr<std::string>> temp_strings_;
  std::vector<char*> runtime_args_;
  std::vector<char*> compiler_args_;
};

}  // namespace

extern "C" int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree-run-mlir");

  // Initialize the compiler once on startup before using any other APIs.
  ireeCompilerGlobalInitialize();

  // Parse full argument list and split into compiler/runtime flag sets.
  ArgParser arg_parser;
  if (!arg_parser.Parse(argc, argv)) {
    ireeCompilerGlobalShutdown();
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(EXIT_FAILURE);
    return EXIT_FAILURE;
  }

  // Pass along compiler flags.
  // Since this is a command line tool we initialize the global compiler
  // command line environment prior to processing the sources.
  // In-process/library uses would usually not do this and would set session
  // specific arguments as needed from whatever configuration mechanisms they
  // use (kwargs passed to python functions, etc).
  ireeCompilerSetupGlobalCL(arg_parser.compiler_argc(),
                            arg_parser.compiler_argv(), "iree-run-mlir",
                            /*installSignalHandlers=*/true);

  // Pass along runtime flags.
  // Note that positional args are left in runtime_argv (after progname).
  // Runtime flags are generally only useful in command line tools where there's
  // a fixed set of devices, a short lifetime, a single thread, and a single
  // context/set of modules/etc. Hosting applications can programmatically
  // do most of what the flags do in a way that avoids the downsides of such
  // global one-shot configuration.
  int runtime_argc = arg_parser.runtime_argc();
  char** runtime_argv = arg_parser.runtime_argv();
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &runtime_argc,
                           &runtime_argv);

  // Ensure a source file was found.
  if (runtime_argc != 2) {
    fprintf(stderr,
            "ERROR: one source MLIR file must be specified.\n"
            "Pass either the path to a .mlir/mlirbc file or `-` to read from "
            "stdin.\n");
    fflush(stderr);
    IREE_TRACE_ZONE_END(z0);
    IREE_TRACE_APP_EXIT(EXIT_FAILURE);
    return EXIT_FAILURE;
  }
  const char* source_filename = runtime_argv[1];

  // Sessions can be reused for many compiler invocations.
  iree_compiler_session_t* session = ireeCompilerSessionCreate();

  // The process return code is 0 for success and non-zero otherwise.
  // We don't differentiate between compiler or runtime error codes here but
  // could if someone found it useful.
  int exit_code = EXIT_SUCCESS;

  // Compile and run the provided source file and get the exit code determined
  // based on the run mode.
  auto status_or = CompileAndRunFile(session, source_filename);
  if (status_or.ok()) {
    exit_code = status_or.value();
  } else {
    exit_code = 2;
    iree_status_fprint(stderr, status_or.status().get());
    fflush(stderr);
  }

  ireeCompilerSessionDestroy(session);

  // No more compiler APIs can be called after this point.
  ireeCompilerGlobalShutdown();

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}

}  // namespace iree
