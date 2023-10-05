# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Collection, Dict, Union
import functools
from pathlib import Path
from tqdm import tqdm
import urllib.parse
import urllib.request


def show_progress(t):
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


@functools.cache
def get_artifact_root_dir() -> Path:
    # TODO: Make configurable.
    return Path.cwd() / "artifacts"


class ArtifactGroup:
    """A group of artifacts with a persistent location on disk."""

    _INSTANCES: Dict[str, "ArtifactGroup"] = {}

    def __init__(self, group_name: str):
        self.group_name = group_name
        if group_name:
            self.directory = get_artifact_root_dir() / group_name
        else:
            self.directory = get_artifact_root_dir()
        self.directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get(cls, group: Union["ArtifactGroup", str]) -> "ArtifactGroup":
        if isinstance(group, ArtifactGroup):
            return group
        try:
            return cls._INSTANCES[group]
        except KeyError:
            instance = ArtifactGroup(group)
            cls._INSTANCES[group] = instance
            return instance


class Artifact:
    """Some form of artifact materialized to disk."""

    def __init__(
        self,
        group: Union[ArtifactGroup, str],
        name: str,
        depends: Collection["Artifact"] = (),
    ):
        self.group = ArtifactGroup.get(group)
        self.name = name
        self.depends = tuple(depends)

    @property
    def path(self) -> Path:
        return self.group.directory / self.name

    def join(self):
        """Waits for the artifact to become available."""
        pass

    def __str__(self):
        return str(self.path)


class ProducedArtifact(Artifact):
    def __init__(
        self,
        group: Union[ArtifactGroup, str],
        name: str,
        callback: Callable[["ProducedArtifact"], Any],
        *,
        always_produce: bool = False,
        depends: Collection["Artifact"] = (),
    ):
        self.group = ArtifactGroup.get(group)
        super().__init__(group, name, depends)
        self.name = name
        self.callback = callback
        self.always_produce = always_produce

    @property
    def stamp_path(self) -> Path:
        """Path of a stamp file which indicates successful transfer."""
        return self.path.with_suffix(self.path.suffix + ".stamp")

    def start(self) -> "ProducedArtifact":
        if not self.always_produce and self.stamp_path.exists():
            if self.path.exists():
                print(f"Not producing {self} because it has already been produced")
                return self
            self.stamp_path.unlink()
        self.callback(self)
        if not self.path.exists():
            raise RuntimeError(
                f"Artifact {self} succeeded generation but was not produced"
            )
        self.stamp()
        return self

    def stamp(self):
        self.stamp_path.touch()


class FetchedArtifact(ProducedArtifact):
    """Represents an artifact that is to be fetched."""

    def __init__(self, group: Union[ArtifactGroup, str], url: str):
        name = Path(urllib.parse.urlparse(url).path).name
        super().__init__(group, name, FetchedArtifact._callback)
        self.url = url

    @staticmethod
    def _callback(self: "FetchedArtifact"):
        print(f"Downloading {self.url} -> {self.path}", flush=True, end="")
        with tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=str(self.path),
        ) as t:
            urllib.request.urlretrieve(self.url, self.path, reporthook=show_progress(t))
        print(f": Retrieved {self.path.stat().st_size} bytes")


class StreamArtifact(Artifact):
    def __init__(self, group: Union[ArtifactGroup, str], name: str):
        super().__init__(group, name)
        self.io = open(self.path, "ab", buffering=0)

    def __del__(self):
        self.io.close()

    def write_line(self, line: Union[str, bytes]):
        contents = line if isinstance(line, bytes) else line.encode()
        self.io.write(contents + b"\n")
