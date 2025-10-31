#!/bin/bash

# Include this script in your shell PATH to enable argument completion for IREE
# tools.

_iree_tools_autocomplete() {
    local tool="$1"
    local commands
    local cur

    # Get the list of commands from the appropriate tool's `--help-list`
    commands=$($tool --help-list | awk '{print $1}' | grep -v '^ *=' | sed 's/[=<].*//')

    cur=${COMP_WORDS[COMP_CWORD]}

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
