# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Helper classes for assembling sets of FX modules that can be compiled.

This uses the `torch.export` machinery. However, it provides some extra
services for handling multiple modules, save/load, and state management.
"""

import functools
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from iree.importers.torch.decompositions import current_aot_decompositions

__all__ = ["FxPrograms", "FxProgramsBuilder", "SharedStateTensor"]


class FxPrograms:
    """Represents a named set of ExportedPrograms.

    This facility works around a design flaw in Torch where they conflated
    ExportedPrograms as representing a single entry-point while also having
    each instance persist its own state_dict and constants. How many times,
    in how many frameworks, do we have to fight this design flaw? Apparently
    once more.

    This base class represents the set of programs, either loaded from storage
    or built live. The tricky part it is managing is to do all of this while
    aliasing state and captured constants. Having those be physically shared
    is an essential optimization.

    In order to manage saving/loading of the set of things, we manually splice
    the state_dict and constants dict such that while saving, we only persist
    the first encountered instance of any reference. Any subsequent instances
    are replaced with a SharedStateTensor, which on load can be re-associated.

    As this is primarily targeted at being able to decouple FX tracing from
    further manipulation (which for reasons unknown, is competing with the
    race of entropy to the heat death of the universe in terms of performance),
    we don't take a lot of pains to be optimized for distribution or storage of
    the resulting artifacts.

    In the future, this same technique could be employed to elide parameters
    that we know we are going to resolve symbolically later, keeping them from
    being loaded and consuming memory during model export and compilation.

    We have faith that in the fullness of time, the design flaws in Torch that
    require this kind of thing to exist will be resolved, and we then won't
    need this hack.
    """

    def __init__(self):
        self.programs: dict[str, torch.export.ExportedProgram] = {}

    def save(self, path: str | os.PathLike) -> int:
        """Saves the set of exported programs to a descriptor file.

        Returns the number of tensors deduped (for debugging/testing).
        """
        path = Path(path).resolve()

        def permute_path(name):
            return path.parent / f"{path.stem}_{name}.pt2"

        # Assemble descriptor.
        program_files = {name: str(permute_path(name)) for name in self.programs.keys()}
        descriptor = {
            "load_order": list(program_files.keys()),
            "program_files": program_files,
        }

        # Accumulate shared state as we go.
        shared_state_dict: dict[str, Any] = {}
        shared_constants: dict[str, Any] = {}
        count_deduped = 0

        # Save each.
        for program_name, export_def in self.programs.items():
            ep = export_def
            assert isinstance(ep, torch.export.ExportedProgram)
            # First validate the ep with normal rules, which we will then
            # disable since we are violating the spec.
            ep._validate()
            orig_state_dict = dict(ep.state_dict)
            constants_dict = ep.constants
            orig_constants = dict(constants_dict)

            try:
                # Now unmerge the state_dict and constants by knocking it up against
                # our running shared state dict.
                count_deduped += _sharify_state_dict(shared_state_dict, ep.state_dict)
                count_deduped += _sharify_state_dict(shared_constants, constants_dict)

                # And save our hacked program.
                save_path = program_files[program_name]
                torch.export.save(ep, save_path)
            finally:
                ep.state_dict.clear()
                ep.state_dict.update(orig_state_dict)
                constants_dict.clear()
                constants_dict.update(orig_constants)

        # Save the descriptor.
        with open(path, "w") as f:
            json.dump(descriptor, f)
        return count_deduped

    @staticmethod
    def load(path: str | os.PathLike) -> "FxPrograms":
        instance = FxPrograms()
        path = Path(path).resolve()
        with open(path, "rb") as f:
            descriptor = json.load(f)

        shared_state_dict: dict[str, Any] = {}
        shared_constants: dict[str, Any] = {}

        for program_name in descriptor["load_order"]:
            program_file_name = descriptor["program_files"][program_name]
            ep = torch.export.load(path.parent / program_file_name)
            _unsharify_state_dict(shared_state_dict, ep.state_dict)
            _unsharify_state_dict(shared_constants, ep.constants)
            instance.programs[program_name] = ep
        return instance


class FxProgramsBuilder(FxPrograms):
    """Builds a new set of exported programs that are all variations of the
    same root nn.Module.

    This can be used to construct multi-entrypoint sets of ExportedPrograms
    in a way that alias information is preserved for lifted tensors.

    Usage:

    ```
    class MyModule(nn.Module):
        ...

    fxb = FxProgramBuilder(MyModule())

    @fxb.export_program(args=example_args)
    def entrypoint(m, x, y):
        return m.forward(x, y)

    fxb.save("/some/path.json")
    ```
    """

    def __init__(self, root_module: nn.Module):
        super().__init__()
        self.root_module = root_module

    def export_program(
        fx_builder,
        f=None,
        *,
        args=None,
        kwargs=None,
        dynamic_shapes=None,
        strict: bool = True,
        name: str | None = None,
    ):
        if f is None:
            return functools.partial(
                fx_builder.export_program,
                args=args,
                kwargs=kwargs,
                strict=strict,
                dynamic_shapes=dynamic_shapes,
                name=name,
            )

        if name is None:
            name = f.__name__
        if name in fx_builder.programs:
            raise ValueError(f"Attempt to export program '{name}' multiple times")

        class LambdaModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.add_module("root", fx_builder.root_module)

        # Here we do a tricky thing: The free-function that we take has
        # signature:
        #   def free_function(root_module, arg1, *, kwarg1)
        # Since the export machinery expects to be able to inspect and query
        # based on user-specified argument names ("arg1", "kwarg1" above),
        # we use the usual @functools.wraps to copy metadata. Because we wrap
        # it before adding it to the class, the first-arg of the free function
        # ("root_module" above) lines up with the usual "self" arg of a method
        # attached to a class. When instantiated and created, this synthetic
        # 'forward' method will inspect as only taking the user-specified
        # argument names (i.e. "arg1", "kwarg1") because the class machinery
        # swallowed the first, which is exactly the one we wanted to elide
        # from Dynamo's view anyway.
        # If we weren't doing this, we would need to munge the signature
        # descriptors to line up because the export machinery needs to see
        # the user-specified function arguments, not our "pseudo-self" root
        # module argument that we always pass.
        # Note that to keep Dynamo happy, we are careful to only access
        # names and attributes in the module tree (vs from the surrounding
        # closure, which goes down less well-trodden paths).
        @functools.wraps(f)
        def new_forward(self, *forward_args, **forward_kwargs):
            return f(self.root, *forward_args, **forward_kwargs)

        setattr(LambdaModule, "forward", new_forward)
        lambda_module = LambdaModule()

        # Export our franken-module.
        extra_kwargs = {}
        if dynamic_shapes:
            extra_kwargs["dynamic_shapes"] = dynamic_shapes
        program = torch.export.export(
            lambda_module, args=args, kwargs=kwargs, strict=strict, **extra_kwargs
        )
        current_decomps = current_aot_decompositions()
        if current_decomps:
            program = program.run_decompositions(current_decomps)

        fx_builder.programs[name] = program
        return program


class SharedStateTensor(torch.Tensor):
    """A fake tensor that we shove into ExportedProgram state to share."""

    @staticmethod
    def __new__(
        cls,
        size,
        dtype,
        shared_state_dict_key: str,
        is_param: bool,
        requires_grad=False,
    ):
        # Using a meta tensor as the wrapped gives us shape and dtype
        # propagation.
        return torch.Tensor._make_subclass(
            cls,
            torch.empty(size, dtype=dtype, device="meta"),
            require_grad=requires_grad,
        )

    def __init__(
        self,
        size,
        dtype,
        shared_state_dict_key: str,
        is_param: bool,
        requires_grad=False,
    ):
        self.shared_state_dict_key = shared_state_dict_key
        # Magic attribute that makes isinstance(t, Parameter) True.
        # See torch.nn.Parameter.
        self._is_param = is_param


def _create_shared_state_tensor(
    like: torch.Tensor, shared_state_dict_key: str
) -> SharedStateTensor:
    t = SharedStateTensor(
        like.size(),
        like.dtype,
        shared_state_dict_key=shared_state_dict_key,
        is_param=isinstance(like, torch.nn.Parameter),
        requires_grad=like.requires_grad,
    )
    return t


def _sharify_state_dict(shared_dict: dict, local_dict: dict) -> int:
    count_deduped = 0
    for key, local_value in local_dict.items():
        if not isinstance(local_value, torch.Tensor):
            continue
        if key in shared_dict:
            shared_value = shared_dict[key]
            assert (
                shared_value is local_value
            ), f"State dict key collision results in different instances ({key})!"
            local_dict[key] = _create_shared_state_tensor(local_value, key)
            count_deduped += 1
        else:
            # Remember the original for the next time.
            shared_dict[key] = local_value
    return count_deduped


def _unsharify_state_dict(shared_dict: dict, local_dict: dict):
    for key, local_value in local_dict.items():
        if not isinstance(local_value, torch.Tensor):
            continue
        if isinstance(local_value, SharedStateTensor):
            # Replace shared state tensor.
            shared_key = local_value.shared_state_dict_key
            try:
                shared_value = shared_dict[shared_key]
            except KeyError:
                raise KeyError(
                    f"Shared tensor not found during deserialization. Corrupt metadata? "
                    f"{shared_key}"
                )
            local_dict[key] = shared_value
        else:
            # Remember this one for later.
            shared_dict[key] = local_value
