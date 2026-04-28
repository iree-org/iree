#!/bin/bash

# Include this script in your shell PATH to enable argument completion for IREE
# tools.

_iree_tools_autocomplete() {
    local tool="$1"
    local commands
    local cur

    cur=${COMP_WORDS[COMP_CWORD]}

    # If the tool is not available yet, avoid noisy shell errors.
    if ! command -v "$tool" >/dev/null 2>&1; then
      case "$cur" in
        -*)
          COMPREPLY=()
          ;;
        *)
          COMPREPLY=($(compgen -f -- "${cur}"))
          ;;
      esac
      return 0
    fi

    # Get the list of options from the tool's `--help-list`.
    commands=$("$tool" --help-list 2>/dev/null | awk '{print $1}' | grep -v '^ *=' | sed 's/[=<].*//')

    case "$cur" in
      -*)
        COMPREPLY=($(compgen -W "$commands" -- "${cur}"))
        ;;
      *)
        COMPREPLY=($(compgen -f -- "${cur}"))
        ;;
    esac

    return 0
}

complete -F _iree_tools_autocomplete iree-compile
complete -F _iree_tools_autocomplete iree-link
complete -F _iree_tools_autocomplete iree-opt
