#!/usr/bin/env python3

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import dataclasses
import json
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ImageInfo:
    """Information of a docker image."""

    deps: List[str]
    digest: Optional[str] = None
    url: Optional[str] = None


def load_image_graph(graph_path: pathlib.Path) -> Dict[str, ImageInfo]:
    """Load image graph from a JSON object."""

    graph_obj = json.loads(graph_path.read_text())
    image_graph = dict((name, ImageInfo(**info)) for name, info in graph_obj.items())
    return image_graph


def dump_image_graph(image_graph: Dict[str, ImageInfo]) -> Dict[str, Any]:
    """Dump image graph into a JSON object."""

    graph_obj = dict(
        (name, dataclasses.asdict(info)) for name, info in image_graph.items()
    )
    return graph_obj


def run_command(
    command: Sequence[str],
    dry_run: bool = False,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
    **run_kwargs,
) -> subprocess.CompletedProcess:
    """Thin wrapper around subprocess.run"""
    print(f"Running: `{' '.join(command)}`")
    if dry_run:
        # Dummy CompletedProess with successful returncode.
        return subprocess.CompletedProcess(command, returncode=0)

    completed_process = subprocess.run(
        command,
        text=text,
        check=check,
        capture_output=capture_output,
        **run_kwargs,
    )
    return completed_process


def check_gcloud_auth(dry_run: bool = False):
    # Ensure the user has the correct authorization if they try to push to GCR.
    try:
        run_command(["which", "gcloud"])
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "gcloud not found. See https://cloud.google.com/sdk/install for "
            "installation."
        ) from error
    run_command(["gcloud", "auth", "configure-docker"], dry_run)
