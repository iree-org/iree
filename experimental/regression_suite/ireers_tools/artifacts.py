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
import os
from azure.storage.blob import BlobClient, BlobProperties
import hashlib
import mmap
import re
import logging

logger = logging.getLogger(__name__)
# Adjust logging levels.
logging.basicConfig(level=logging.INFO)
for log_name, log_obj in logging.Logger.manager.loggerDict.items():
    if log_name.startswith("azure"):
        logging.getLogger(log_name).setLevel(logging.WARNING)


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
    root_path = os.getenv("IREE_TEST_FILES", default=str(Path.cwd())) + "/artifacts"
    return Path(os.path.expanduser(root_path)).resolve()


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
        depends: Collection["Artifact"] = (),
    ):
        self.group = ArtifactGroup.get(group)
        super().__init__(group, name, depends)
        self.name = name
        self.callback = callback

    def start(self) -> "ProducedArtifact":
        self.callback(self)
        if not self.path.exists():
            raise RuntimeError(
                f"Artifact {self} succeeded generation but was not produced"
            )
        return self


class FetchedArtifact(ProducedArtifact):
    """Represents an artifact that is to be fetched."""

    def __init__(self, group: Union[ArtifactGroup, str], url: str):
        name = Path(urllib.parse.urlparse(url).path).name
        super().__init__(group, name, FetchedArtifact._callback)
        self.url = url

    def human_readable_size(self, size, decimal_places=2):
        for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
            if size < 1024.0 or unit == "PiB":
                break
            size /= 1024.0
        return f"{size:.{decimal_places}f} {unit}"

    def get_azure_md5(self, remote_file: str, azure_blob_properties: BlobProperties):
        """Gets the content_md5 hash for a blob on Azure, if available."""
        content_settings = azure_blob_properties.get("content_settings")
        if not content_settings:
            return None
        azure_md5 = content_settings.get("content_md5")
        if not azure_md5:
            logger.warning(
                f"  Remote file '{remote_file}' on Azure is missing the "
                "'content_md5' property, can't check if local matches remote"
            )
        return azure_md5

    def get_local_md5(self, local_file_path: Path):
        """Gets the content_md5 hash for a lolca file, if it exists."""
        if not local_file_path.exists() or local_file_path.stat().st_size == 0:
            return None

        with open(local_file_path) as file, mmap.mmap(
            file.fileno(), 0, access=mmap.ACCESS_READ
        ) as file:
            return hashlib.md5(file).digest()

    def download_azure_artifact(self: "FetchedArtifact"):
        """
        Checks the hashes between the local file and azure file.
        """
        remote_file_name = self.url.rsplit("/", 1)[-1]

        # Extract path components from Azure URL to use with the Azure Storage Blobs
        # client library for Python (https://pypi.org/project/azure-storage-blob/).
        #
        # For example:
        #   https://sharkpublic.blob.core.windows.net/sharkpublic/path/to/blob.txt
        #                                            ^           ^
        #   account_url:    https://sharkpublic.blob.core.windows.net
        #   container_name: sharkpublic
        #   blob_name:      path/to/blob.txt
        result = re.search(r"(https.+\.net)/([^/]+)/(.+)", self.url)
        account_url = result.groups()[0]
        container_name = result.groups()[1]
        blob_name = result.groups()[2]

        with BlobClient(
            account_url,
            container_name,
            blob_name,
            max_chunk_get_size=1024 * 1024 * 32,  # 32 MiB
            max_single_get_size=1024 * 1024 * 32,  # 32 MiB
        ) as blob_client:
            blob_properties = blob_client.get_blob_properties()
            blob_size_str = self.human_readable_size(blob_properties.size)
            azure_md5 = self.get_azure_md5(self.url, blob_properties)

            local_md5 = self.get_local_md5(self.path)

            if azure_md5 and azure_md5 == local_md5:
                logger.info(
                    f"  Skipping '{remote_file_name}' download ({blob_size_str}) "
                    "- local MD5 hash matches"
                )

            if not local_md5:
                logger.info(
                    f"  Downloading '{remote_file_name}' ({blob_size_str}) "
                    f"to '{self.path}'"
                )
                with open(self.path, mode="wb") as local_blob:
                    download_stream = blob_client.download_blob(max_concurrency=4)
                    local_blob.write(download_stream.readall())
            else:
                logger.info(
                    f"  Downloading '{remote_file_name}' ({blob_size_str}) "
                    f"to '{self.path}' (local MD5 does not match)"
                )
                with open(self.path, mode="wb") as local_blob:
                    download_stream = blob_client.download_blob(max_concurrency=4)
                    local_blob.write(download_stream.readall())

    @staticmethod
    def _callback(self: "FetchedArtifact"):
        if "blob.core.windows.net" in self.url:
            self.download_azure_artifact()


class StreamArtifact(Artifact):
    def __init__(self, group: Union[ArtifactGroup, str], name: str):
        super().__init__(group, name)
        self.io = open(self.path, "ab", buffering=0)

    def __del__(self):
        self.io.close()

    def write_line(self, line: Union[str, bytes]):
        contents = line if isinstance(line, bytes) else line.encode()
        self.io.write(contents + b"\n")
