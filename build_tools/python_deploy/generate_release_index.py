#!/usr/bin/env python3

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Creates an HTML page for releases for `pip install --find-links` from GitHub releases.

Sample usage:

    ```bash
    ./build_tools/python_deploy/generate_release_index.py \
        --repos=iree-org/iree,iree-org/iree-turbine \
        --output=docs/website/docs/pip-release-links.html
    ```

WARNING for developers:
  GitHub's APIs have rate limits:
    * 60 requests per hour for unauthenticated users
    * 5000 requests per hour for authenticated users

  This script only requires read access to public endpoints, but you
  authenticate yourself to take advantage of the higher rate limit.
  Creating a fine-grained personal access token with no extra permissions and
  storing it in the `GITHUB_TOKEN` environment variable seems to work well
  enough.

  See documentation at:
    * https://docs.github.com/en/rest/using-the-rest-api/getting-started-with-the-rest-api?apiVersion=2022-11-28#authentication
    * https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api?apiVersion=2022-11-28
    * https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api?apiVersion=2022-11-28
"""

# TODO(#10479) since we're generating this we might as well create a PEP 503
# compliant index

import argparse
import html
import io
import os
import requests
import sys
import textwrap

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repos",
        "--repositories",
        default="iree-org/iree",
        help="Comma-delimited list of GitHub repositories to fetch releases from.",
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

        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if GITHUB_TOKEN:
            self.headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

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
                headers=self.headers,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Request was not successful, reason: {response.reason}"
                )
            for release in response.json():
                yield release
            if "next" not in response.links:
                break
            page += 1


def add_releases_for_repository(repo: str, file: io.TextIOWrapper):
    fetcher = ReleaseFetcher(repo)

    file.write(
        f'    <h2>Packages for <a href="https://github.com/{repo}">{repo}</a></h2>\n'
    )

    for release in fetcher.get_all():
        if release["draft"]:
            continue
        for asset in release["assets"]:
            url = html.escape(asset["browser_download_url"])
            name = html.escape(asset["name"])
            file.write(f"    <a href={url}>{name}</a><br>\n")


def main(args):
    repos = args.repos.split(",")
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
        for repo in repos:
            add_releases_for_repository(repo, f)
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
