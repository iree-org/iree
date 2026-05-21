# AMDGPU Device Binaries

This package builds the small device-side AMDGPU support library embedded into
the HAL runtime. These code objects contain blit kernels and runtime utility
kernels; they are not math libraries, so the build intentionally targets LLVM
generic ISA processors wherever LLVM documents a compatible generic target.

The public selector vocabulary mirrors ROCm/TheRock so users can request the
same target families in both builds. The source of truth for that family naming
is TheRock's
[`cmake/therock_amdgpu_targets.cmake`](https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake).
The source of truth for generic ISA coverage is LLVM's AMDGPU generic processor
documentation and tablegen data under `third_party/llvm-project/llvm/`.

## Support Mechanism

The single checked-in target map lives in
`build_tools/scripts/amdgpu_target_map.py`. It records:

- exact HSA ISA architecture suffixes, such as `gfx1100`;
- the code object target to compile for each exact architecture, such as
  `gfx11-generic`;
- TheRock-style selectors, such as `gfx110X-all`, `dgpu-all`, and `igpu-all`.

Running the script emits small generated fragments:

- `target_map.bzl`, loaded by `targets.bzl` for Bazel selector expansion;
- `target_map.cmake`, included by `CMakeLists.txt` for CMake selector expansion;
- `runtime/src/iree/hal/drivers/amdgpu/util/target_id_map.inl`, included by
  `target_id.c` so runtime ISA lookup uses the same exact-to-code-object map as
  the build.

The generated files are checked in. Pre-commit runs
`python build_tools/scripts/amdgpu_target_map.py --check` so CI catches drift
between the Python map and the generated fragments.

## Regenerating Code Objects

This generator supports the checked-in code object flow where the runtime build
consumes prebuilt blobs instead of building LLVM. Regenerate those code objects
only when the builtin device sources change or when adding/removing an
architecture from the checked-in set:

```bash
python build_tools/scripts/amdgpu_device_binaries.py \
  --output-dir runtime/src/iree/hal/drivers/amdgpu/device/binaries/prebuilt \
  --rocm-path /path/to/rocm-or-therock-sdk \
  --targets gfx9-generic,gfx90a,gfx9-4-generic,gfx10-1-generic,gfx10-3-generic,gfx11-generic,gfx12-generic
```

The script accepts the same selector vocabulary as the build-time target map:
exact targets such as `gfx1100`, code-object targets such as `gfx11-generic`,
and families such as `gfx94X-all`. If `--targets` is omitted, the script uses
`IREE_HAL_AMDGPU_DEVICE_BINARY_TARGETS` when set, otherwise it builds the
checked-in generic-family set plus `gfx90a`. Pass
`--all-targets` to build every known code-object target. Some ROCm releases may
not yet support every generic target recorded in the map; in that case the
script fails before compilation and reports the unsupported target names.

Tool discovery is intentionally compatible with both in-tree and out-of-tree
LLVM flows. Explicit `--clang`, `--llvm-link`, `--lld`, and `--llvm-objcopy`
flags win. Otherwise the script checks individual
`IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_*`, `IREE_*`, and LLVM environment variables,
host-tool directories such as
`IREE_HOST_BIN_DIR`, `IREE_HOST_TOOLS`, `IREE_BINARY_DIR`,
`IREE_LLVM_TOOLS_DIR`, and `LLVM_TOOLS_BINARY_DIR`, then ROCm roots: explicit
`--rocm-path` entries, `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH`,
`IREE_ROCM_PATH`, the configure-style `ROCM_PATH`, `ROCM_ROOT`, `ROCM_HOME`,
`HIP_PATH` variables, a root inferred from `hipcc` on `PATH`, and `/opt/rocm`.
For ROCm installs the script searches standard layouts such as `llvm/bin`,
`lib/llvm/bin`, and `bin`; after finding `clang` or `amdclang` it also asks
that compiler where its companion LLVM tools live with `--print-prog-name`.

By default the generator keeps only the `.kd` kernel descriptor symbols in the
regular symbol tables. Do not replace this with `llvm-strip --strip-all` or
`llvm-objcopy --strip-sections`: ROCr may still load or inspect symbol/section
metadata when resolving builtin kernels by name. The device kernel declarations
carry the liveness contract via `IREE_AMDGPU_ATTRIBUTE_KERNEL`; the generator
and live source builds only provide a local-all version script and remove local
symbols after linking.

## Build Modes

Normal runtime builds use checked-in blobs from `prebuilt/`. The optional Bazel
AMDGPU device toolchain repository defaults to an inert stub and source builds
must opt into a real producer with
`--repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm`, `llvm-tools`,
`llvm-project`, or `auto`. This keeps AMDGPU HAL runtime rebuilds independent
of the LLVM submodule and ROCm tools unless a developer explicitly asks to
rebuild the device code objects from source.

Bazel source mode is enabled with:

```bash
iree-bazel-build \
  --config=amdgpu_device_binaries_source_rocm \
  --repo_env=IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH=/path/to/rocm-or-therock-sdk \
  --//runtime/src/iree/hal/drivers/amdgpu/device/binaries:targets=gfx11-generic \
  //runtime/src/iree/hal/drivers/amdgpu/device/binaries:toc
```

Use `--config=amdgpu_device_binaries_source_llvm_project` when deliberately
building through the in-tree `@llvm-project` repository instead of ROCm tools.
Both Bazel source modes invoke `build_tools/scripts/amdgpu_device_binaries.py`;
the target selector flag accepts the same exact, code-object, and family
selectors described above.

CMake uses the matching cache variables:

- `IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=prebuilt|source`, default
  `prebuilt`.
- `IREE_HAL_AMDGPU_DEVICE_BINARY_TARGETS`, default
  `gfx9-generic;gfx90a;gfx9-4-generic;gfx10-1-generic;gfx10-3-generic;gfx11-generic;gfx12-generic`.
- `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=auto|rocm|llvm-tools|llvm-project`, default
  `auto` for source mode.
- `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH` for a ROCm or TheRock SDK root.
- `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_TOOLS_DIR` for a directory containing
  `clang`, `llvm-link`, `lld`, and `llvm-objcopy`.
- `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_BINARY`,
  `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_LINK_BINARY`,
  `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLD_BINARY`,
  `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_LLVM_OBJCOPY_BINARY`, and
  `IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_CLANG_RESOURCE_INCLUDE` for exact per-tool
  overrides.

In source mode CMake invokes `build_tools/scripts/amdgpu_device_binaries.py`.
The `rocm` and `auto` modes search standard ROCm installs, ask `clang` or
`amdclang` for companion tool locations, check `hipcc` on `PATH`, and check
`/opt/rocm`. The `llvm-tools` mode searches explicit LLVM/IREE tool directories
and per-tool overrides. The `llvm-project` mode uses tools already configured by
the containing IREE build.

## Current Generic-Family Audit

The current map intentionally includes generic code-object coverage for the
modern families LLVM documents and the ROCm/TheRock selector vocabulary names:

| Selector family | Exact targets | Code-object target |
| --- | --- | --- |
| `gfx9` GCN | `gfx900`, `gfx902`, `gfx904`, `gfx906`, `gfx909`, `gfx90c` | `gfx9-generic` |
| `gfx9-4` CDNA | `gfx940`, `gfx941`, `gfx942`, `gfx950` | `gfx9-4-generic` |
| `gfx10.1` RDNA | `gfx1010`, `gfx1011`, `gfx1012`, `gfx1013` | `gfx10-1-generic` |
| `gfx10.3` RDNA | `gfx1030`, `gfx1031`, `gfx1032`, `gfx1033`, `gfx1034`, `gfx1035`, `gfx1036` | `gfx10-3-generic` |
| `gfx11` RDNA/APU | `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, `gfx1152`, `gfx1153`, `gfx1170`, `gfx1171`, `gfx1172` | `gfx11-generic` |
| `gfx12` RDNA | `gfx1200`, `gfx1201` | `gfx12-generic` |
| `gfx12.5` RDNA | `gfx1250`, `gfx1251` | `gfx12-5-generic` |

`gfx12-5-generic` is available as an explicit selector, but it is intentionally
not part of the default checked-in prebuilt set until ROCm/LLVM can link these
builtin kernels for that generic target.

Targets outside this table should fail selection loudly until LLVM documents
their code-object compatibility and the embedded support library has been
compiled for the required exact or generic processor.

## Adding An Architecture

When a new AMDGPU architecture is supported:

1. Check TheRock's `therock_amdgpu_targets.cmake` for the selector/family
   vocabulary and product-family membership.
2. Check LLVM's AMDGPU generic processor docs/tablegen files for the matching
   generic ISA target. If LLVM does not document generic coverage for the new
   exact ISA, keep the code object exact until that support exists.
3. Update `EXACT_TARGET_CODE_OBJECTS` and `TARGET_FAMILIES` in
   `build_tools/scripts/amdgpu_target_map.py`.
4. Run `python build_tools/scripts/amdgpu_target_map.py`.
5. Run `buildifier`, `clang-format`, the target-map pre-commit check, and the
   focused AMDGPU device-binary build/test targets.

Do not hand-edit `target_map.bzl`, `target_map.cmake`, or `target_id_map.inl`.
Do not add a parallel table in `device_library.c`; the runtime loader consumes
`target_id` helpers directly.
