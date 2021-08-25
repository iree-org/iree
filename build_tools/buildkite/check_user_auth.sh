#!/bin/bash

set -eu
set -o pipefail

BUILDKITE_USER_IS_AUTHORIZED=0

if [[ "${BUILDKITE_BUILD_CREATOR_EMAIL}" == *@google.com ]]; then
  echo "User '${BUILDKITE_BUILD_CREATOR_EMAIL}' is authorized because email ends in 'google.com'"
  BUILDKITE_USER_IS_AUTHORIZED=1
  exit 0
else
  echo "User email '${BUILDKITE_BUILD_CREATOR_EMAIL}' does not end in google.com"
fi

if grep -q "${BUILDKITE_BUILD_CREATOR_EMAIL}" build_tools/buildkite/authorized_user_emails.txt; then
  echo "User '${BUILDKITE_BUILD_CREATOR_EMAIL}' is authorized because they are listed in authorized_user_emails.txt"
  BUILDKITE_USER_IS_AUTHORIZED=1
  exit 0
else
  echo "User email '${BUILDKITE_BUILD_CREATOR_EMAIL}' is not listed in authorized_user_emails.txt"
fi

echo "User is not authorized"
