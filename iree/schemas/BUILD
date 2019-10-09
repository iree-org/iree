load("//third_party/flatbuffers:build_defs.bzl", "flatbuffer_cc_library")
load("//tools/build_defs/cc:cc_embed_data.bzl", "cc_embed_data")
load("//tools/build_rules:build_test.bzl", "build_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

FLATC_ARGS = [
    # Preserve workspace-relative include paths in generated code.
    "--keep-prefix",
    # Use C++11 'enum class' for enums.
    "--scoped-enums",
    # Include reflection tables used for dumping debug representations.
    "--reflect-names",
    # Generate FooT types for unpack/pack support. Note that this should only
    # be used in tooling as the code size/runtime overhead is non-trivial.
    "--gen-object-api",
]

flatbuffer_cc_library(
    name = "archive_def_cc_fbs",
    srcs = ["archive_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
        ":bytecode_def_cc_fbs_includes",
        ":device_def_cc_fbs_includes",
        ":device_group_def_cc_fbs_includes",
        ":device_table_def_cc_fbs_includes",
        ":executable_def_cc_fbs_includes",
        ":executable_table_def_cc_fbs_includes",
        ":function_def_cc_fbs_includes",
        ":function_table_def_cc_fbs_includes",
        ":module_def_cc_fbs_includes",
        ":source_map_def_cc_fbs_includes",
        ":type_def_cc_fbs_includes",
    ],
)

flatbuffer_cc_library(
    name = "bytecode_def_cc_fbs",
    srcs = ["bytecode_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
    ],
)

flatbuffer_cc_library(
    name = "debug_service_cc_fbs",
    srcs = ["debug_service.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
        ":bytecode_def_cc_fbs_includes",
        ":device_def_cc_fbs_includes",
        ":device_group_def_cc_fbs_includes",
        ":device_table_def_cc_fbs_includes",
        ":executable_def_cc_fbs_includes",
        ":executable_table_def_cc_fbs_includes",
        ":function_def_cc_fbs_includes",
        ":function_table_def_cc_fbs_includes",
        ":module_def_cc_fbs_includes",
        ":source_map_def_cc_fbs_includes",
        ":type_def_cc_fbs_includes",
    ],
)

flatbuffer_cc_library(
    name = "device_def_cc_fbs",
    srcs = ["device_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
    ],
)

flatbuffer_cc_library(
    name = "device_group_def_cc_fbs",
    srcs = ["device_group_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
    ],
)

flatbuffer_cc_library(
    name = "device_table_def_cc_fbs",
    srcs = ["device_table_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
        ":device_def_cc_fbs_includes",
        ":device_group_def_cc_fbs_includes",
    ],
)

flatbuffer_cc_library(
    name = "executable_def_cc_fbs",
    srcs = ["executable_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
    ],
)

flatbuffer_cc_library(
    name = "executable_table_def_cc_fbs",
    srcs = ["executable_table_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
        ":executable_def_cc_fbs_includes",
    ],
)

flatbuffer_cc_library(
    name = "function_def_cc_fbs",
    srcs = ["function_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
        ":bytecode_def_cc_fbs_includes",
        ":type_def_cc_fbs_includes",
    ],
)

flatbuffer_cc_library(
    name = "function_table_def_cc_fbs",
    srcs = ["function_table_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
        ":bytecode_def_cc_fbs_includes",
        ":function_def_cc_fbs_includes",
        ":type_def_cc_fbs_includes",
    ],
)

flatbuffer_cc_library(
    name = "module_def_cc_fbs",
    srcs = ["module_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
        ":bytecode_def_cc_fbs_includes",
        ":device_def_cc_fbs_includes",
        ":device_group_def_cc_fbs_includes",
        ":device_table_def_cc_fbs_includes",
        ":executable_def_cc_fbs_includes",
        ":executable_table_def_cc_fbs_includes",
        ":function_def_cc_fbs_includes",
        ":function_table_def_cc_fbs_includes",
        ":source_map_def_cc_fbs_includes",
        ":type_def_cc_fbs_includes",
    ],
)

flatbuffer_cc_library(
    name = "source_map_def_cc_fbs",
    srcs = ["source_map_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
    ],
)

flatbuffer_cc_library(
    name = "spirv_executable_def_cc_fbs",
    srcs = ["spirv_executable_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
    ],
)

flatbuffer_cc_library(
    name = "type_def_cc_fbs",
    srcs = ["type_def.fbs"],
    flatc_args = FLATC_ARGS,
    gen_reflections = True,
    includes = [
    ],
)

build_test(
    name = "schema_build_test",
    targets = [
        ":archive_def_cc_fbs",
        ":bytecode_def_cc_fbs",
        ":debug_service_cc_fbs",
        ":device_def_cc_fbs",
        ":device_group_def_cc_fbs",
        ":device_table_def_cc_fbs",
        ":executable_def_cc_fbs",
        ":executable_table_def_cc_fbs",
        ":function_def_cc_fbs",
        ":function_table_def_cc_fbs",
        ":module_def_cc_fbs",
        ":source_map_def_cc_fbs",
        ":spirv_executable_def_cc_fbs",
        ":type_def_cc_fbs",
    ],
)

cc_library(
    name = "schemas",
    hdrs = [
        ":archive_def_generated.h",
        ":bytecode_def_generated.h",
        ":debug_service_generated.h",
        ":device_def_generated.h",
        ":device_group_def_generated.h",
        ":device_table_def_generated.h",
        ":executable_def_generated.h",
        ":executable_table_def_generated.h",
        ":function_def_generated.h",
        ":function_table_def_generated.h",
        ":module_def_generated.h",
        ":source_map_def_generated.h",
        ":type_def_generated.h",
    ],
    deps = [
        ":archive_def_cc_fbs",
        ":bytecode_def_cc_fbs",
        ":debug_service_cc_fbs",
        ":device_def_cc_fbs",
        ":device_group_def_cc_fbs",
        ":device_table_def_cc_fbs",
        ":executable_def_cc_fbs",
        ":executable_table_def_cc_fbs",
        ":function_def_cc_fbs",
        ":function_table_def_cc_fbs",
        ":module_def_cc_fbs",
        ":source_map_def_cc_fbs",
        ":spirv_executable_def_cc_fbs",
        ":type_def_cc_fbs",
        "@flatbuffers",
    ],
)

cc_embed_data(
    name = "reflection_data",
    srcs = [
        "archive_def.bfbs",
        "bytecode_def.bfbs",
        "debug_service.bfbs",
        ":device_def.bfbs",
        ":device_group_def.bfbs",
        ":device_table_def.bfbs",
        "executable_def.bfbs",
        "executable_table_def.bfbs",
        "function_def.bfbs",
        "function_table_def.bfbs",
        "module_def.bfbs",
        "source_map_def.bfbs",
        "spirv_executable_def.bfbs",
        "type_def.bfbs",
    ],
    outs = [
        "reflection_data.cc",
        "reflection_data.h",
        "reflection_data_bin.o",
    ],
    embedopts = ["--namespace=iree::schemas"],
)
