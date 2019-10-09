# IREE Debugger UIs.
#
# The main debugger UI can be used in standalone mode connected to a remote
# host (via :debugger) or can be directly embedded into the IREE runtime to
# allow for attaching (--iree_attach_debugger).
#
# By default the IREE runtime does not compile in debug support. To link it in
# pass --define=IREE_DEBUG=1 to bazel builds of the runtime.

load("//third_party/emscripten:split_transition_defs.bzl", "auto_wasm_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

alias(
    name = "debugger",
    actual = select({
        "//tools/cc_target_os:emscripten": ":debug_app_emscripten_files",
        "//conditions:default": ":debug_app_native",
    }),
)

cc_library(
    name = "debug_app_library",
    srcs = ["debug_app.cc"],
    hdrs = ["debug_app.h"],
    deps = [
        "//iree/base:memory",
        "//iree/base:source_location",
        "//iree/base:status",
        "//iree/schemas",
        "//iree/vm:bytecode_printer",
        "//iree/vm:bytecode_tables_sequencer",
        "//iree/vm:module",
        "//iree/vm:source_map",
        "//iree/vm/debug:debug_client",
        "//third_party/GL:GLES2_headers",
        "//third_party/SDL2",
        "//third_party/absl/flags:flag",
        "//third_party/absl/memory",
        "//third_party/absl/strings",
        "//third_party/absl/types:optional",
        "//third_party/dear_imgui",
        "//third_party/dear_imgui:imgui_sdl_opengl3",
    ],
)

# NOTE: users must also link in a GL implementation, like:
#        "//third_party/GL/native:GLESv2",  # build-cleaner: keep
cc_library(
    name = "debug_app_embedded",
    srcs = ["debug_app_embedded.cc"],
    hdrs = ["debug_app_embedded.h"],
    deps = [
        ":debug_app_library",
        "//iree/base:memory",
        "//iree/base:status",
        "//third_party/SDL2",
        "//third_party/absl/base:core_headers",
        "//third_party/absl/memory",
        "//third_party/absl/strings",
        "//third_party/absl/synchronization",
        "//third_party/dear_imgui",
    ],
)

EMSCRIPTEN_LINKOPTS_COMMON = [
    # Error at compile time on unresolved symbols.
    "-s ERROR_ON_UNDEFINED_SYMBOLS=1",

    # Required by SDL.
    "-s EXTRA_EXPORTED_RUNTIME_METHODS=Pointer_stringify",

    # TODO(benvanik): tweak to enable support when needed.
    "-s ALLOW_MEMORY_GROWTH=1",
    # "-s WASM_MEM_MAX=268435456",  # 256MB
    # "-s TOTAL_MEMORY=268435456",  # 256MB
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

cc_binary(
    name = "debug_app_emscripten",
    srcs = ["debug_app_main_emscripten.cc"],
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
        ":debug_app_library",
        "//iree/base:init",
        "//iree/base:source_location",
        "//iree/base:status",
        "//third_party/SDL2",
        "//third_party/absl/memory",
        "//third_party/dear_imgui",
        "//third_party/dear_imgui:imgui_sdl_opengl3",
    ],
)

auto_wasm_binary(
    name = "debug_app_emscripten_binary",
    cc_target = ":debug_app_emscripten",
    tags = ["manual"],
)

Fileset(
    name = "debug_app_emscripten_files",
    out = "wasm_files",
    entries = [
        FilesetEntry(
            files = [":debug_app_emscripten_binary"],
            strip_prefix = "debug_app_emscripten_binary",
            destdir = "wasm",
        ),
        FilesetEntry(
            files = ["debug_app.html"],
            destdir = "wasm",
        ),
    ],
    tags = ["manual"],
)

cc_binary(
    name = "debug_app_native",
    srcs = ["debug_app_main_native.cc"],
    deps = [
        ":debug_app_embedded",
        "//iree/base:init",
        "//iree/base:status",
        "//third_party/GL/native:EGL",  # build-cleaner: keep
        "//third_party/GL/native:GLESv2",  # build-cleaner: keep
    ],
)

cc_binary(
    name = "debug_cli",
    srcs = ["debug_cli_main.cc"],
    deps = [
        ":debug_prompt",
        "//iree/base:init",
        "//iree/base:status",
        "//third_party/absl/flags:flag",
    ],
)

cc_library(
    name = "debug_prompt",
    srcs = ["debug_prompt.cc"],
    hdrs = ["debug_prompt.h"],
    deps = [
        "//iree/base:status",
        "//iree/vm/debug:debug_client",
        "//third_party/absl/strings",
    ],
)
