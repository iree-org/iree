#!/bin/bash
# Asserts that a genquery output file is empty (no results).
# Used by iree_assert_no_dependency to verify that a forbidden transitive
# dependency is not present.
if [ -s "$1" ]; then
  echo "ERROR: Forbidden dependency detected:"
  cat "$1"
  exit 1
fi
