# IREE Debugger UIs.
#
# The main debugger UI can be used in standalone mode connected to a remote
# host (via :debugger) or can be directly embedded into the IREE runtime to
# allow for attaching (--iree_attach_debugger).
#
# By default the IREE runtime does not compile in debug support. To link it in
# pass --define=IREE_DEBUG=1 to bazel builds of the runtime.

# TODO(benvanik): re-enable debugger after refactoring.
# load("//third_party/emscripten:split_transition_defs.bzl", "auto_wasm_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(benvanik): re-enable debugger after refactoring.
# alias(
#     name = "debugger",
#     actual = select({
#         "//tools/cc_target_os:emscripten": ":debug_app_emscripten_files",
#         "//conditions:default": ":debug_app_native",
#     }),
# )
#
# cc_library(
#     name = "debug_app_library",
#     srcs = ["debug_app.cc"],
#     hdrs = ["debug_app.h"],
#     deps = [
#         "//third_party/GL:GLES2_headers",
#         "//third_party/SDL2",
#         "@com_google_absl//absl/flags:flag",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/types:optional",
#         "//third_party/dear_imgui",
#         "//third_party/dear_imgui:imgui_sdl_opengl3",
#         "///iree/base:memory",
#         "///iree/base:source_location",
#         "///iree/base:status",
#         "///iree/rt/debug:debug_client",
#         "///iree/schemas",
#     ],
# )
#
# # NOTE: users must also link in a GL implementation, like:
# #        "//third_party/GL/native:GLESv2",  # build-cleaner: keep
# cc_library(
#     name = "debug_app_embedded",
#     srcs = ["debug_app_embedded.cc"],
#     hdrs = ["debug_app_embedded.h"],
#     deps = [
#         ":debug_app_library",
#         "//third_party/SDL2",
#         "@com_google_absl//absl/base:core_headers",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/synchronization",
#         "//third_party/dear_imgui",
#         "///iree/base:memory",
#         "///iree/base:status",
#     ],
# )
#
# EMSCRIPTEN_LINKOPTS_COMMON = [
#     # Error at compile time on unresolved symbols.
#     "-s ERROR_ON_UNDEFINED_SYMBOLS=1",
#
#     # Required by SDL.
#     "-s EXTRA_EXPORTED_RUNTIME_METHODS=Pointer_stringify",
#
#     # TODO(benvanik): tweak to enable support when needed.
#     "-s ALLOW_MEMORY_GROWTH=1",
#     # "-s WASM_MEM_MAX=268435456",  # 256MB
#     # "-s TOTAL_MEMORY=268435456",  # 256MB
# ]
#
# EMSCRIPTEN_LINKOPTS_DBG = [
#     # Show WASM stack trace in Chrome debugger.
#     "-g2",
#     "-s DEMANGLE_SUPPORT=1",
#
#     # Enable verbose assertions.
#     "-s ASSERTIONS=2",
#     "-s SAFE_HEAP=1",
#     "-s STACK_OVERFLOW_CHECK=2",
# ]
#
# EMSCRIPTEN_LINKOPTS_OPT = []
#
# cc_binary(
#     name = "debug_app_emscripten",
#     srcs = ["debug_app_main_emscripten.cc"],
#     linkopts = EMSCRIPTEN_LINKOPTS_COMMON + select({
#         "//tools/compilation_mode:dbg": EMSCRIPTEN_LINKOPTS_DBG,
#         "//tools/compilation_mode:opt": EMSCRIPTEN_LINKOPTS_OPT,
#         "//conditions:default": EMSCRIPTEN_LINKOPTS_OPT,
#     }),
#     tags = [
#         "manual",
#         "notap",  # TODO(b/137088911): Build/test on TAP
#         "wasm",
#     ],
#     deps = [
#         ":debug_app_library",
#         "//third_party/SDL2",
#         "@com_google_absl//absl/memory",
#         "//third_party/dear_imgui",
#         "//third_party/dear_imgui:imgui_sdl_opengl3",
#         "///iree/base:init",
#         "///iree/base:source_location",
#         "///iree/base:status",
#     ],
# )
#
# auto_wasm_binary(
#     name = "debug_app_emscripten_binary",
#     cc_target = ":debug_app_emscripten",
#     tags = ["manual"],
# )
#
# Fileset(
#     name = "debug_app_emscripten_files",
#     out = "wasm_files",
#     entries = [
#         FilesetEntry(
#             files = [":debug_app_emscripten_binary"],
#             strip_prefix = "debug_app_emscripten_binary",
#             destdir = "wasm",
#         ),
#         FilesetEntry(
#             files = ["debug_app.html"],
#             destdir = "wasm",
#         ),
#     ],
#     tags = ["manual"],
# )
#
# cc_binary(
#     name = "debug_app_native",
#     srcs = ["debug_app_main_native.cc"],
#     deps = [
#         ":debug_app_embedded",
#         "//third_party/GL/native:EGL",  # build-cleaner: keep
#         "//third_party/GL/native:GLESv2",  # build-cleaner: keep
#         "///iree/base:init",
#         "///iree/base:status",
#     ],
# )
#
# cc_binary(
#     name = "debug_cli",
#     srcs = ["debug_cli_main.cc"],
#     deps = [
#         ":debug_prompt",
#         "@com_google_absl//absl/flags:flag",
#         "///iree/base:init",
#         "///iree/base:status",
#     ],
# )
#
# cc_library(
#     name = "debug_prompt",
#     srcs = ["debug_prompt.cc"],
#     hdrs = ["debug_prompt.h"],
#     deps = [
#         "@com_google_absl//absl/strings",
#         "///iree/base:status",
#         "///iree/rt/debug:debug_client",
#     ],
# )
