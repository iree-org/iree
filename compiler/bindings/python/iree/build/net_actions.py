# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import urllib.error
import urllib.request

from iree.build.executor import BuildAction, BuildContext, BuildFile, BuildFileMetadata

__all__ = [
    "fetch_http",
]


def fetch_http(*, name: str, url: str) -> BuildFile:
    context = BuildContext.current()
    output_file = context.allocate_file(name)
    action = FetchHttpAction(
        url=url, output_file=output_file, desc=f"Fetch {url}", executor=context.executor
    )
    output_file.deps.add(action)
    return output_file


class FetchHttpAction(BuildAction):
    def __init__(self, url: str, output_file: BuildFile, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.output_file = output_file
        self.original_desc = self.desc

    def _invoke(self):
        # Determine whether metadata indicates that fetch is needed.
        path = self.output_file.get_fs_path()
        needs_fetch = False
        existing_metadata = self.output_file.access_metadata()
        existing_url = existing_metadata.get("fetch_http.url")
        if existing_url != self.url:
            needs_fetch = True

        # Always fetch if empty or absent.
        if not path.exists() or path.stat().st_size == 0:
            needs_fetch = True

        # Bail if already obtained.
        if not needs_fetch:
            return

        # Download to a staging file.
        stage_path = path.with_name(f".{path.name}.download")
        self.executor.write_status(f"Fetching URL: {self.url} -> {path}")

        def reporthook(received_blocks: int, block_size: int, total_size: int):
            received_size = received_blocks * block_size
            if total_size == 0:
                self.desc = f"{self.original_desc} ({received_size} bytes received)"
            else:
                complete_percent = round(100 * received_size / total_size)
                self.desc = f"{self.original_desc} ({complete_percent}% complete)"

        try:
            urllib.request.urlretrieve(self.url, str(stage_path), reporthook=reporthook)
        except urllib.error.HTTPError as e:
            raise IOError(f"Failed to fetch URL '{self.url}': {e}") from None
        finally:
            self.desc = self.original_desc

        # Commit the download.
        def commit(metadata: BuildFileMetadata) -> bool:
            metadata["fetch_http.url"] = self.url
            path.unlink(missing_ok=True)
            stage_path.rename(path)
            return True

        self.output_file.access_metadata(commit)
