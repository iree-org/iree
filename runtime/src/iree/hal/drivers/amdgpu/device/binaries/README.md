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

## Current Generic-Family Audit

The current map intentionally includes generic code-object coverage for the
modern families LLVM documents and the ROCm/TheRock selector vocabulary names:

| Selector family | Exact targets | Code-object target |
| --- | --- | --- |
| `gfx9-4` CDNA | `gfx940`, `gfx941`, `gfx942`, `gfx950` | `gfx9-4-generic` |
| `gfx11` RDNA/APU | `gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`, `gfx1150`, `gfx1151`, `gfx1152`, `gfx1153`, `gfx1170`, `gfx1171`, `gfx1172` | `gfx11-generic` |
| `gfx12` RDNA | `gfx1200`, `gfx1201` | `gfx12-generic` |
| `gfx12.5` RDNA | `gfx1250`, `gfx1251` | `gfx12-5-generic` |

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
   focused AMDGPU device-library build/test targets.

Do not hand-edit `target_map.bzl`, `target_map.cmake`, or `target_id_map.inl`.
Do not add a parallel table in `device_library.c`; the runtime loader consumes
`target_id` helpers directly.
