#!/bin/bash
set -e

BINARY="$1"
shift
ARGS="$@"

echo "=== memcheck ==="
compute-sanitizer --tool=memcheck --leak-check=full "$BINARY" $ARGS

echo "=== racecheck ==="
compute-sanitizer --tool=racecheck "$BINARY" $ARGS

echo "=== synccheck ==="
compute-sanitizer --tool=synccheck "$BINARY" $ARGS
