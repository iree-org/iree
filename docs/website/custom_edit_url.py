#!/usr/bin/env python3
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Custom edit URL plugin for mkdocs.

MkDocs can create links to view/edit the source of a website page in the
associated source location (e.g. GitHub repository).

* https://www.mkdocs.org/user-guide/configuration/#edit_uri
* https://squidfunk.github.io/mkdocs-material/setup/adding-a-git-repository/#code-actions

For markdown files in the documentation folder, this is a straightforward
process, taking the relative path and appending it to the source location:
  /repo_root/docs/section/page.md -->
  https://github.com/[org]/[repo]/blob/main/docs/section/page.md

For generated files, the inferred URL does not match the actual source:
  /repo_root/src/file.cc -->
  /repo_root/docs/gen/file.md -->
  [broken link] https://github.com/[org]/[repo]/blob/main/docs/gen/file.md

This plugin allows for pages to explicitly specify their source URL via
markdown frontmatter.

References:
* https://github.com/renovatebot/renovatebot.github.io/pull/187
* https://github.com/mkdocs/mkdocs/discussions/2757

Usage:

1. Add a hook to mkdocs.yml by following
    https://www.mkdocs.org/user-guide/configuration/#hooks

2. Add frontmatter to any pages you want to customize:

  ---
  custom_edit_url: [full hyperlink to the source location, e.g. https://github.com/...]
  ---
"""


def on_page_context(context, page, config, **kwargs):
    if "custom_edit_url" in page.meta:
        page.edit_url = page.meta["custom_edit_url"]
    return context
