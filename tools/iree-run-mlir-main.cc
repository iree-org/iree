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
// this can be overridden by passing the --iree-hal-target-device= and related
// flags explicitly. Likewise if only the target backend is specified the
// devices to use will be inferred unless explicitly specified.
//
// Example usage to compile and run with CUDA:
// $ iree-run-mlir --device=cuda://0 file.mlir
// or to compile with the LLVM CPU backend and run with the local-task driver:
// $ iree-run-mlir file.mlir \
//       --Xcompiler,iree-hal-target-device=local \
//       --Xcompiler,iree-hal-local-target-device-backends=llvm-cpu \
//       --device=local-task
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
//       --iree-hal-target-device=local \
//       --iree-hal-local-target-device-backends=llvm-cpu
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
#include "iree/base/tooling/flags.h"
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

// Tries to guess a default runtime device from the |target_device| when
// possible. Users are still able to override this by passing --device= flags.
std::string InferDefaultDeviceFromTargetDevice(
    std::string_view target_device) {
  if (target_device.empty()) {
    return "";
  } else if (target_device == "local") {
    // Local targets default to the multithreaded task system driver.
    return "local-task";
  }

  // Most target device names correspond directly to runtime driver names.
  return std::string(target_device);
}

// Tries to guess a compiler target device from the given runtime |device_uri|.
// Returns an empty string if no target device can be inferred.
std::string InferTargetDeviceFromDevice(iree_string_view_t device_uri) {
  iree_string_view_t driver = iree_string_view_empty();
  iree_string_view_split(device_uri, ':', &driver, nullptr);

  if (iree_string_view_is_empty(driver)) {
    return "";
  } else if (iree_string_view_starts_with(driver, IREE_SV("local-"))) {
    return "local";
  }

  return std::string(driver.data, driver.size);
}

// Tries to infer compiler target devices from the runtime device configuration.
// Returns a comma-delimited list of target devices.
StatusOr<std::string> InferTargetDevicesFromDevices(
    iree_string_view_list_t device_uris) {
  if (device_uris.count == 0) return "";

  std::set<std::string> target_devices;
  for (iree_host_size_t i = 0; i < device_uris.count; ++i) {
    auto target_device = InferTargetDeviceFromDevice(device_uris.values[i]);
    if (!target_device.empty()) {
      target_devices.insert(std::move(target_device));
    }
  }

  std::string result;
  for (auto& target_device : target_devices) {
    if (!result.empty()) result.append(",");
    result.append(target_device);
  }

  return result;
}

// Configures compiler target device flags based on the --device= flags set by
// the user, or infers a default runtime device from an explicitly specified
// compiler target device. Online compilers may want to perform more
// sophisticated device selection and target configuration.
Status ConfigureTargetDevices(iree_compiler_session_t* session,
                              std::string* out_default_device_uri) {
  // Query explicitly specified compiler target devices.
  std::string target_devices_flag;
  ireeCompilerSessionGetFlags(
      session, /*nonDefaultOnly=*/true,
      [](const char* flag_str, size_t length, void* user_data) {
        std::string_view prefix = "--iree-hal-target-device=";
        std::string_view flag = std::string_view(flag_str, length);
        if (starts_with(prefix, flag)) {
          flag.remove_prefix(prefix.size());
          if (flag.empty()) return;
          auto* result = static_cast<std::string*>(user_data);
          *result = std::string(flag);
        }
      },
      static_cast<void*>(&target_devices_flag));

  iree_string_view_list_t device_uris = iree_hal_device_flag_list();

  // Nothing was explicitly configured.
  if (target_devices_flag.empty() && device_uris.count == 0) {
    return OkStatus();
  }

  // Both sides were explicitly configured.
  if (!target_devices_flag.empty() && device_uris.count > 0) {
    return OkStatus();
  }

  // Compiler target device specified but no runtime device: infer one when
  // there is a single simple target device.
  if (device_uris.count == 0) {
    if (target_devices_flag.find(',') != std::string::npos) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "if multiple target devices are specified the runtime device to use "
          "must also be specified with --device= (have "
          "`--iree-hal-target-device=%.*s`)",
          (int)target_devices_flag.size(), target_devices_flag.data());
    }

    *out_default_device_uri =
        InferDefaultDeviceFromTargetDevice(target_devices_flag);
    return OkStatus();
  }

  // Runtime devices were specified but compiler target devices were not.
  IREE_ASSIGN_OR_RETURN(auto target_devices,
                        InferTargetDevicesFromDevices(device_uris));

  if (!target_devices.empty()) {
    std::vector<std::string> compiler_flags;
    compiler_flags.push_back(
        std::string("--iree-hal-target-device=") + target_devices);

    // Preserve the historical default for local runtime devices: compile with
    // llvm-cpu unless the user explicitly configured compiler flags.
    if (target_devices == "local" ||
        target_devices.find("local,") != std::string::npos ||
        target_devices.find(",local") != std::string::npos) {
      compiler_flags.push_back(
          "--iree-hal-local-target-device-backends=llvm-cpu");
    }

    std::vector<const char*> compiler_argv;
    compiler_argv.reserve(compiler_flags.size());
    for (const auto& flag : compiler_flags) {
      compiler_argv.push_back(flag.c_str());
    }

    auto error = ireeCompilerSessionSetFlags(
        session, compiler_argv.size(), compiler_argv.data());
    if (error) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "unable to set inferred target device configuration");
    }
  }

  return OkStatus();
}

// Runs the given .mlir file based on the current flags.
StatusOr<int> CompileAndRunFile(iree_compiler_session_t* session,
                                const char* mlir_filename) {
  IREE_TRACE_SCOPE_NAMED("CompileAndRunFile");

  // Configure compiler target devices and/or get the default runtime device if
  // either side was not explicitly specified.
  // Note that compiler target devices and runtime devices aren't necessarily
  // 1:1 and this is an imperfect guess. In this simple online compiler we assume
  // homogeneous device sets, while library/hosting layers can configure more
  // complex target and runtime device configurations.
  std::string default_device_uri;
  IREE_RETURN_IF_ERROR(ConfigureTargetDevices(session, &default_device_uri));

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
      iree_status_t status = iree_make_status(status_code, "%s", msg);
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

static int main(int argc, char** argv) {
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

int main(int argc, char** argv) { return iree::main(argc, argv); }
