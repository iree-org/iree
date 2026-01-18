---
icon: material/bug
---

# Fuzzing with libFuzzer

[libFuzzer](https://llvm.org/docs/LibFuzzer.html) is a coverage-guided fuzzing
engine provided by LLVM. It generates random inputs and mutates them based on
code coverage feedback to find crashes, hangs, and memory errors.

IREE provides build infrastructure for creating libFuzzer-based fuzz targets
that integrate with the existing build system.

## When to use fuzzing

Fuzzing is most effective for:

- Parsers and decoders (UTF-8, binary formats, etc.)
- Serialization/deserialization code
- Input validation logic
- Any code that processes untrusted or external data

## Enabling fuzzing builds

### Bazel

```shell
bazel build --config=fuzzer //runtime/src/iree/base/internal:unicode_fuzz
```

The `--config=fuzzer` flag enables coverage instrumentation and ASan.

### CMake

```shell
cmake -B build -DIREE_ENABLE_FUZZING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target unicode_fuzz
```

Fuzzing automatically enables ASan. Fuzz targets are excluded from the default
`all` target and must be built explicitly.

## Running fuzz targets

Fuzz targets are standalone executables that accept libFuzzer arguments:

```shell
# Run indefinitely (Ctrl+C to stop)
./build/runtime/src/iree/base/internal/unicode_fuzz

# Run for 60 seconds
./build/runtime/src/iree/base/internal/unicode_fuzz -max_total_time=60

# Use a corpus directory (recommended)
mkdir -p corpus/unicode
./build/runtime/src/iree/base/internal/unicode_fuzz corpus/unicode/
```

### Common options

Option | Description
------ | -----------
`-max_total_time=N` | Stop after N seconds
`-max_len=N` | Maximum input size in bytes
`-timeout=N` | Per-input timeout in seconds (0 to disable)
`-jobs=N` | Run N parallel fuzzing jobs
`-workers=N` | Number of worker processes for parallel fuzzing
`-dict=file` | Use a dictionary file for structured inputs
`-seed=N` | Use specific random seed for reproducibility

See [libFuzzer documentation](https://llvm.org/docs/LibFuzzer.html) for all
options.

## Writing fuzz targets

Fuzz targets implement the `LLVMFuzzerTestOneInput` function:

```cpp
// my_fuzz.cc
#include <stddef.h>
#include <stdint.h>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Process the fuzzer-generated input
  my_function_under_test(data, size);
  return 0;  // Always return 0
}
```

### Adding to the build system

In `BUILD.bazel`:

```python
load("//build_tools/bazel:build_defs.oss.bzl", "iree_runtime_cc_fuzz")

iree_runtime_cc_fuzz(
    name = "my_fuzz",
    srcs = ["my_fuzz.cc"],
    deps = [
        ":my_library",
    ],
)
```

Then run `python build_tools/bazel_to_cmake/bazel_to_cmake.py` to generate the
CMake equivalent.

## Best practices

### Maintain a corpus

Store interesting inputs in a corpus directory. The fuzzer uses existing corpus
entries as seeds for mutation:

```shell
mkdir -p corpus/my_fuzz
./my_fuzz corpus/my_fuzz/ -max_total_time=3600
```

After finding bugs, minimize the corpus to remove redundant entries:

```shell
mkdir corpus/my_fuzz_minimized
./my_fuzz -merge=1 corpus/my_fuzz_minimized/ corpus/my_fuzz/
```

### Add unit tests for found bugs

When fuzzing discovers a crash:

1. Minimize the reproducer: `./my_fuzz -minimize_crash=1 crash-xxx`
2. Add the minimized input as a unit test case
3. Fix the bug
4. Verify the fix with the original crash input

This prevents regressions and documents the bug.

### Use dictionaries for structured formats

For inputs with specific syntax (protocols, file formats), provide a dictionary:

```text
# my_dict.txt
"keyword1"
"keyword2"
"\x00\x01\x02"
```

```shell
./my_fuzz -dict=my_dict.txt corpus/
```

## Troubleshooting

### Fuzzer runs slowly

- Ensure `CMAKE_BUILD_TYPE=RelWithDebInfo` or `Release` (Debug is very slow)
- Check that the target doesn't do excessive I/O or allocations per iteration
- Use `-jobs=N` for parallel fuzzing on multi-core machines

### Out of memory

- Limit input size with `-max_len=N`
- Add early returns for oversized inputs in your fuzz target
- Use `-rss_limit_mb=N` to set memory limits

### No new coverage

- Verify the target actually processes the input
- Check that coverage instrumentation is enabled (`-fsanitize=fuzzer-no-link`)
- Try seeding with representative inputs in the corpus

### Timeout errors

libFuzzer kills inputs that take too long (default 1200 seconds). If you see
`ALARM: working on the last Unit for N seconds` followed by a timeout:

- Use `-timeout=N` to adjust the per-input timeout (in seconds)
- Use `-timeout=0` to disable timeouts entirely (useful for debugging)
- Check if certain inputs cause algorithmic complexity issues (e.g., pathological
  regex patterns, deeply nested structures)
