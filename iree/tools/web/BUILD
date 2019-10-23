# IREE web tools.

load("//third_party/emscripten:split_transition_defs.bzl", "auto_wasm_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

EMSCRIPTEN_LINKOPTS_COMMON = [
    # Error at compile time on unresolved symbols.
    "-s ERROR_ON_UNDEFINED_SYMBOLS=1",

    # Note: If pthreads and memory growth are enabled, WASM_MEM_MAX must be set.
    # Also, USE_PTHREADS + ALLOW_MEMORY_GROWTH may run non-wasm code slowly.
    # "-s ALLOW_MEMORY_GROWTH=1",
    # "-s WASM_MEM_MAX=268435456",  # 256MB
    # "-s TOTAL_MEMORY=268435456",  # 256MB

    # Request a prepopulated pool of web workers for pthreads to use.
    # Without this, threads may not start until the javascript thread yields.
    # See considerations at https://emscripten.org/docs/porting/pthreads.html.
    "-s PTHREAD_POOL_SIZE=1",
]

EMSCRIPTEN_LINKOPTS_DBG = [
    # Show WASM stack trace in Chrome debugger.
    "-g2",
    "-s DEMANGLE_SUPPORT=1",

    # Enable verbose assertions.
    "-s ASSERTIONS=2",
    "-s SAFE_HEAP=1",
    "-s STACK_OVERFLOW_CHECK=2",
]

EMSCRIPTEN_LINKOPTS_OPT = []

# To use run_module_emscripten:
#  > bazel build third_party/iree/tools/web:run_module_emscripten_files

cc_binary(
    name = "run_module_emscripten",
    srcs = ["run_module_emscripten.cc"],
    linkopts = EMSCRIPTEN_LINKOPTS_COMMON + select({
        "//tools/compilation_mode:dbg": EMSCRIPTEN_LINKOPTS_DBG,
        "//tools/compilation_mode:opt": EMSCRIPTEN_LINKOPTS_OPT,
        "//conditions:default": EMSCRIPTEN_LINKOPTS_OPT,
    }),
    tags = [
        "manual",
        "notap",  # TODO(b/137088911): Build/test on TAP
        "wasm",
    ],
    deps = [
        "///iree/base:init",
        "///iree/base:status",
        "///iree/hal:buffer_view_string_util",
        "///iree/hal:driver_registry",
        "///iree/hal/interpreter:interpreter_driver_module",
        "///iree/rt",
        "///iree/vm:sequencer_module",
        "//third_party/emscripten:embind",
    ],
)

auto_wasm_binary(
    name = "run_module_emscripten_binary",
    cc_target = ":run_module_emscripten",
    tags = ["manual"],
    threads = "emscripten",
)

Fileset(
    name = "run_module_emscripten_files",
    out = "wasm_files",
    entries = [
        FilesetEntry(
            files = [":run_module_emscripten_binary"],
            strip_prefix = "run_module_emscripten_binary",
            destdir = "wasm",
        ),
        FilesetEntry(
            files = ["run_module.html"],
            destdir = "wasm",
        ),
    ],
    tags = ["manual"],
)
