#!/bin/bash

# Start an X server via Xvfb, if one does not exist already

set -e

add_line() {
    if grep -q "^$2\$" "$1"; then
        echo "String '$2' already exists in '$1'; not adding"
    else
        echo "Adding line '$2' to '$1'"
        echo "$2" >> "$1"
    fi
}

if [ ! -z "$(pgrep '^Xvfb\b')" ]; then
    echo 'Xvfb already running; will skip'
else
    echo 'Starting (or restarting) Xvfb'
    for job in $(pgrep '^Xvfb\b'); do kill $job; done \
        && rm -f ~/.Xauthority \
        && Xvfb -screen 0 640x480x16 -nolisten tcp -auth ~/.Xauthority \
                -maxclients 2048 :0 &
    disown
    echo 'Done, use display :0'
fi
