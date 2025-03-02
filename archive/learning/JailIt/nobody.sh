#!/usr/bin/env bash
#
# nobody.sh

#!/bin/bash

# Check if application was provided
if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/application"
  exit 1
fi

APP_PATH=$1
APP_NAME=$(basename "$APP_PATH")

# Directory for the jail
JAIL=~/jail

# Create the jail directory
mkdir -p $JAIL

# List of directories to create in the jail
DIRS=(bin etc lib lib64 sbin usr var Applications)

# Create directories
for DIR in "${DIRS[@]}"; do
  mkdir -p "$JAIL/$DIR"
done

# List of binaries (and their dependencies) to copy to the jail
BINARIES=(/bin/bash /bin/ls /usr/bin/id /usr/bin/whoami)

# Copy binaries and dependencies
for BINARY in "${BINARIES[@]}"; do
  cp "$BINARY" "$JAIL$BINARY"

  # Get a list of dependencies for this binary
  DEPENDENCIES=$(otool -L "$BINARY" | awk 'NR>1 {print $1}')

  # Copy each dependency
  for DEPENDENCY in $DEPENDENCIES; do
    if [[ "$DEPENDENCY" != /System/* ]]; then
      cp "$DEPENDENCY" "$JAIL$DEPENDENCY"
    fi
  done
done

# Copy the application into the jail
cp -R "$APP_PATH" "$JAIL/Applications/"

# Chroot into the jail and run the application
sudo chroot $JAIL /Applications/"$APP_NAME"

# Clean up the jail
rm -rf $JAIL

