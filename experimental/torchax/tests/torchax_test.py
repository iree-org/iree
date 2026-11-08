# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
import sys
import textwrap
import unittest


def _run_alone(script):
    """`script` in a fresh process, without the key preamble."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
    )


class TorchaxImportTest(unittest.TestCase):
    def test_import_loads_neither_torchax_nor_jax(self):
        result = _run_alone(
            """
            import sys
            import torch

            backend = torch._C._get_privateuse1_backend_name()
            import iree_torchax

            assert "torchax" not in sys.modules
            assert "jax" not in sys.modules
            assert torch._C._get_privateuse1_backend_name() == backend

            class Affine(torch.nn.Module):
                def forward(self, x):
                    return x * 2 + 1

            torch.export.export(Affine(), (torch.ones(3),), strict=True)
            iree_torchax.exported_program_to_jax
            assert "torchax" in sys.modules
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_child_export_preserves_parent_torch_state(self):
        result = _run_alone(
            """
            import pathlib
            import subprocess
            import sys
            import tempfile
            import torch

            class Affine(torch.nn.Module):
                def forward(self, x):
                    return x * 2 + 1

            backend = torch._C._get_privateuse1_backend_name()
            with tempfile.TemporaryDirectory() as directory:
                output = pathlib.Path(directory) / "model.mlir"
                child = (
                    "import pathlib, sys, torch, iree_torchax; "
                    "Affine = type('Affine', (torch.nn.Module,), "
                    "{'forward': lambda self, x: x * 2 + 1}); "
                    "exported = torch.export.export(Affine(), (torch.ones(3),)); "
                    "_, artifact = iree_torchax.exported_program_to_stablehlo(exported); "
                    "pathlib.Path(sys.argv[1]).write_text(str(artifact.mlir_module()))"
                )
                subprocess.run([sys.executable, "-c", child, str(output)], check=True)
                assert 'func.func public @main' in output.read_text()

            assert "torchax" not in sys.modules
            assert "jax" not in sys.modules
            assert torch._C._get_privateuse1_backend_name() == backend
            torch.export.export(Affine(), (torch.ones(3),), strict=True)
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_registration_loads_neither_torchax_nor_jax(self):
        result = _run_alone(
            """
            import sys
            import torch
            import iree_torchax

            @torch.library.custom_op("iree_lazy_test::twice", mutates_args=())
            def twice(x: torch.Tensor) -> torch.Tensor:
                return x * 2

            @twice.register_fake
            def _(x):
                return torch.empty_like(x)

            op = torch.ops.iree_lazy_test.twice.default
            iree_torchax.register_composite(
                op, lambda x, *, env: x, provider="example.lazy")
            iree_torchax.register_decomposition(
                torch.ops.aten.hardtanh.default, lambda x, lo, hi: x,
                provider="example.lazy")
            sdpa = torch.ops.aten.scaled_dot_product_attention.default
            try:
                iree_torchax.register_composite(
                    sdpa, lambda *a, env: a[0], provider="example.lazy")
            except ValueError as error:
                assert "iree.attention" in str(error), error
            else:
                raise AssertionError("an IREE built-in was replaced silently")
            assert "torchax" not in sys.modules
            assert "jax" not in sys.modules

            class Affine(torch.nn.Module):
                def forward(self, x):
                    return x * 2 + 1

            torch.export.export(Affine(), (torch.ones(3),), strict=True)
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)


_STATE = textwrap.dedent(
    """
    import jax
    import torch
    import torchax
    from torchax.ops import ops_registry
    import iree_torchax

    def state():
        return (
            bool(jax.config.jax_enable_x64),
            jax.config.jax_default_matmul_precision,
            dict(torchax.default_env()._ops),
            dict(ops_registry.all_aten_ops),
            dict(ops_registry.all_torch_functions),
            torch._C._is_torch_function_mode_enabled(),
        )

    class Attend(torch.nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    rows = tuple(torch.randn(1, 2, 3, 4) for _ in range(3))
    attend = torch.export.export(Attend(), rows)
    """
)


class TorchaxScopedExportTest(unittest.TestCase):
    def test_failed_export_restores_the_caller_state(self):
        result = _run_alone(
            _STATE
            + textwrap.dedent(
                """
            @torch.library.custom_op("iree_scope_test::fails", mutates_args=())
            def fails(x: torch.Tensor) -> torch.Tensor:
                return x

            @fails.register_fake
            def _(x):
                return torch.empty_like(x)

            def broken(x, *, env):
                raise RuntimeError("handler failed")

            iree_torchax.register_composite(
                torch.ops.iree_scope_test.fails.default, broken, provider="test")

            class Fails(torch.nn.Module):
                def forward(self, x):
                    return torch.ops.iree_scope_test.fails(x)

            exported = torch.export.export(Fails(), (torch.ones(3),))
            before = state()
            try:
                iree_torchax.exported_program_to_stablehlo(exported)
            except RuntimeError as error:
                assert "handler failed" in str(error), error
            else:
                raise AssertionError("the export did not fail")
            assert state() == before
                """
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_export_restores_the_caller_state(self):
        result = _run_alone(
            _STATE
            + textwrap.dedent(
                """
            before = state()
            _, artifact = iree_torchax.exported_program_to_stablehlo(attend)
            text = str(artifact.mlir_module())
            assert 'composite "torch.scaled_dot_product_attention"' in text
            assert state() == before
                """
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_export_inside_an_environment_leaves_it_active(self):
        result = _run_alone(
            _STATE
            + textwrap.dedent(
                """
            caller = torchax.tensor.Environment()
            outside = state()
            with caller:
                inside = state()
                iree_torchax.exported_program_to_stablehlo(attend)
                assert state() == inside
                ones = torch.ones(2, device="jax")
                assert isinstance(ones + 1, torchax.tensor.Tensor)
            assert state() == outside
                """
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class TorchaxLegacyImportTest(unittest.TestCase):
    def test_export_keeps_default_environment_overrides(self):
        # Exporting through the family modules leaves the default environment
        # as the caller set it.
        result = _run_alone(
            """
            import torch
            aten = torch.ops.aten
            import torchax
            from iree_torchax import torchax_export
            env = torchax.default_env()
            sdpa = aten.scaled_dot_product_attention.default

            def mine(*args, **kwargs):
                raise AssertionError("not called")

            env.override_op_definition(sdpa, mine)

            class Attend(torch.nn.Module):
                def forward(self, q, k, v):
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v)

            rows = tuple(torch.randn(1, 2, 3, 4) for _ in range(3))
            torchax_export.exported_program_to_jax(
                torch.export.export(Attend(), rows))
            assert env._ops[sdpa].func is mine
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class TorchaxExportBoundaryTest(unittest.TestCase):
    def test_export_with_dropout_diagnoses(self):
        result = _run_alone(
            """
            import torch
            import iree_torchax

            class AttendWithDropout(torch.nn.Module):
                def forward(self, query, key, value):
                    return torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.5)

            rows = tuple(torch.randn(1, 2, 3, 4) for _ in range(3))
            exported = torch.export.export(AttendWithDropout(), rows)
            try:
                iree_torchax.exported_program_to_stablehlo(exported)
            except NotImplementedError as error:
                import re
                assert re.search("scaled_dot_product_attention.*dropout", str(error)), error
            else:
                raise AssertionError("dropout export did not fail")
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
