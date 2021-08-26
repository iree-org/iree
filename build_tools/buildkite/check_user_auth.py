#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

AUTHORIZED_USER_EMAILS=["gcmn@google.com", "marius.brehler@iml.fraunhofer.de"]
CREATOR_EMAIL_ENV_VAR="BUILDKITE_BUILD_CREATOR_EMAIL"

def main():
  creator_email = os.getenv(CREATOR_EMAIL_ENV_VAR)
  if not creator_email:
    raise RuntimeError(f"{CREATOR_EMAIL_ENV_VAR} is not defined")

  if creator_email.endswith("@google.com"):
    print(f"User '{creator_email}' is authorized because email ends in '@google.com'")
    return 0
  print(f"User '{creator_email}' does not end in '@google.com'")

  if creator_email in AUTHORIZED_USER_EMAILS:
    print(f"User '{creator_email}' is authorized because email is in list of authorized users")
    return 0
  print(f"User '{creator_email}' is not in authorized user list: {AUTHORIZED_USER_EMAILS}")

  print("User is not authorized")
  return 1

if __name__ == "__main__":
  sys.exit(main())
