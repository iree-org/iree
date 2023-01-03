#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from unittest import mock
import http.client
import requests
import unittest

import post_benchmark_comment


class PostBenchmarkCommentTest(unittest.TestCase):

  def test_post_to_gist(self):
    gist_url = "https://example.com/123455/1234.md"
    response = mock.create_autospec(requests.Response)
    response.status_code = http.client.CREATED
    response.json.return_value = {"html_url": gist_url, "truncated": False}
    requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    requester.post.return_value = response

    url = post_benchmark_comment.post_to_gist(requester=requester,
                                              filename="1234.md",
                                              content="xyz")

    self.assertEqual(url, gist_url)
    requester.post.assert_called_once_with(
        endpoint=post_benchmark_comment.GITHUB_GIST_API,
        payload={
            "public": True,
            "files": {
                "1234.md": {
                    "content": "xyz"
                }
            }
        })

  def test_post_to_gist_truncated(self):
    gist_url = "example.com/123455/1234.md"
    response = mock.create_autospec(requests.Response)
    response.status_code = http.client.CREATED
    response.json.return_value = {"html_url": gist_url, "truncated": True}
    requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    requester.post.return_value = response

    self.assertRaises(
        RuntimeError, lambda: post_benchmark_comment.post_to_gist(
            requester=requester, filename="1234.md", content="xyz"))

  def test_get_previous_comment_on_pr(self):
    response = mock.create_autospec(requests.Response)
    response.status_code = http.client.OK
    response.json.return_value = [{
        "id": 1,
        "user": {
            "login": "bot"
        },
        "body": "comment id: abcd"
    }, {
        "id": 2,
        "user": {
            "login": "user"
        },
        "body": "comment id: 1234"
    }, {
        "id": 3,
        "user": {
            "login": "bot"
        },
        "body": "comment id: 1234"
    }]
    requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    requester.get.return_value = response

    comment_id = post_benchmark_comment.get_previous_comment_on_pr(
        requester=requester,
        pr_number=23,
        gist_bot_user="bot",
        comment_type_id="1234")

    self.assertEqual(comment_id, 3)
    requester.get.assert_called_once_with(
        endpoint=
        f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/23/comments",
        payload={
            "per_page": 100,
            "page": 1,
            "sort": "updated",
            "direction": "desc"
        })

  def test_get_previous_comment_on_pr_not_found(self):
    response = mock.create_autospec(requests.Response)
    response.status_code = http.client.OK
    response.json.return_value = []
    requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    requester.get.return_value = response

    comment_id = post_benchmark_comment.get_previous_comment_on_pr(
        requester=requester,
        pr_number=23,
        gist_bot_user="bot",
        comment_type_id="1234")

    self.assertIsNone(comment_id)
    requester.get.assert_any_call(
        endpoint=
        f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/23/comments",
        payload={
            "per_page": 100,
            "page": 1,
            "sort": "updated",
            "direction": "desc"
        })
    requester.get.assert_any_call(
        endpoint=
        f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/23/comments",
        payload={
            "per_page": 100,
            "page": post_benchmark_comment.MAX_PAGES_TO_SEARCH_PREVIOUS_COMMENT,
            "sort": "updated",
            "direction": "desc"
        })

  def test_update_comment_on_pr(self):
    response = mock.create_autospec(requests.Response)
    response.status_code = http.client.OK
    requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    requester.patch.return_value = response

    post_benchmark_comment.update_comment_on_pr(requester=requester,
                                                comment_id=123,
                                                content="xyz")

    requester.patch.assert_called_once_with(
        endpoint=
        f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/comments/123",
        payload={"body": "xyz"})

  def test_create_comment_on_pr(self):
    response = mock.create_autospec(requests.Response)
    response.status_code = http.client.CREATED
    requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    requester.post.return_value = response

    post_benchmark_comment.create_comment_on_pr(requester=requester,
                                                pr_number=1234,
                                                content="xyz")

    requester.post.assert_called_once_with(
        endpoint=
        f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/1234/comments",
        payload={"body": "xyz"})


if __name__ == "__main__":
  unittest.main()
