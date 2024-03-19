#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Creates an HTML page for releases for `pip install --find-links` from GitHub releases."""
# TODO(#10479) since we're generating this we might as well create a PEP 503
# compliant index
import argparse
import html
import json
import subprocess
import sys
import textwrap

import requests


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        "--repository",
        default="openxla/iree",
        help="The GitHub repository to fetch releases from.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="The file to write the HTML to or '-' for stdout (the default)",
    )
    return parser.parse_args()


class ReleaseFetcher:
    def __init__(self, repo, per_page=100):
        self._session = requests.Session()
        self._repo = repo
        self._per_page = per_page

    def get_all(self):
        url = f"https://api.github.com/repos/{self._repo}/releases"
        page = 1

        # GitHub limits API responses to the first 1000 results.
        while page * self._per_page < 1000:
            response = self._session.get(
                url,
                params={
                    "page": page,
                    "per_page": self._per_page,
                },
            )
            for release in response.json():
                yield release
            if "next" not in response.links:
                break
            page += 1


def main(args):
    fetcher = ReleaseFetcher(repo=args.repo)
    with sys.stdout if args.output == "-" else open(args.output, "w") as f:
        f.write(
            textwrap.dedent(
                """\
            <!DOCTYPE html>
            <html>
              <body>
            """
            )
        )
        for release in fetcher.get_all():
            if release["draft"]:
                continue
            for asset in release["assets"]:
                url = html.escape(asset["browser_download_url"])
                name = html.escape(asset["name"])
                f.write(f"    <a href={url}>{name}</a><br />\n")
        f.write(
            textwrap.dedent(
                """\
      </body>
    </html>
    """
            )
        )


if __name__ == "__main__":
    main(parse_arguments())
