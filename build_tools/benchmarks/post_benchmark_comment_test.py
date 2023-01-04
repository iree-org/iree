#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import http.client
import requests
import unittest
from unittest import mock
from typing import Any

import post_benchmark_comment


class GithubClientTest(unittest.TestCase):

  def test_post_to_gist(self):
    gist_url = "https://example.com/123455/1234.md"
    response = mock.create_autospec(requests.Response)
    response.status_code = http.client.CREATED
    response.json.return_value = {"html_url": gist_url, "truncated": False}
    requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    requester.post.return_value = response
    client = post_benchmark_comment.GithubClient(requester)

    url = client.post_to_gist(filename="1234.md", content="xyz")

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
    client = post_benchmark_comment.GithubClient(requester)

    with self.assertRaises(RuntimeError) as _:
      client.post_to_gist(filename="1234.md", content="xyz")

  def test_get_previous_comment_on_pr(self):
    first_response = mock.create_autospec(requests.Response)
    first_response.status_code = http.client.OK
    first_response.json.return_value = [{
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
    }]
    second_response = mock.create_autospec(requests.Response)
    second_response.status_code = http.client.OK
    second_response.json.return_value = [{
        "id": 3,
        "user": {
            "login": "bot"
        },
        "body": "comment id: 1234"
    }]

    def _handle_get(endpoint: str, payload: Any):
      if payload["page"] == 1:
        return first_response
      if payload["page"] == 2:
        return second_response
      raise ValueError("Unexpected page")

    requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    requester.get.side_effect = _handle_get
    client = post_benchmark_comment.GithubClient(requester)

    comment_id = client.get_previous_comment_on_pr(pr_number=23,
                                                   gist_bot_user="bot",
                                                   comment_type_id="1234")

    self.assertEqual(comment_id, 3)
    endpoint_url = f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/23/comments"
    requester.get.assert_any_call(endpoint=endpoint_url,
                                  payload={
                                      "per_page": 100,
                                      "page": 1,
                                      "sort": "updated",
                                      "direction": "desc"
                                  })
    requester.get.assert_any_call(endpoint=endpoint_url,
                                  payload={
                                      "per_page": 100,
                                      "page": 2,
                                      "sort": "updated",
                                      "direction": "desc"
                                  })

  def test_get_previous_comment_on_pr_not_found(self):
    response = mock.create_autospec(requests.Response)
    response.status_code = http.client.OK
    response.json.return_value = []
    requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    requester.get.return_value = response
    client = post_benchmark_comment.GithubClient(requester)

    comment_id = client.get_previous_comment_on_pr(pr_number=23,
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
    client = post_benchmark_comment.GithubClient(requester)

    client.update_comment_on_pr(comment_id=123, content="xyz")

    requester.patch.assert_called_once_with(
        endpoint=
        f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/comments/123",
        payload={"body": "xyz"})

  def test_create_comment_on_pr(self):
    response = mock.create_autospec(requests.Response)
    response.status_code = http.client.CREATED
    requester = mock.create_autospec(post_benchmark_comment.APIRequester)
    requester.post.return_value = response
    client = post_benchmark_comment.GithubClient(requester)

    client.create_comment_on_pr(pr_number=1234, content="xyz")

    requester.post.assert_called_once_with(
        endpoint=
        f"{post_benchmark_comment.GITHUB_IREE_API_PREFIX}/issues/1234/comments",
        payload={"body": "xyz"})


if __name__ == "__main__":
  unittest.main()
