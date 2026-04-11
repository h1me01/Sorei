#!/bin/bash
set -e

BINARY=./build/debug/examples/astra/astra_example
ARGS="${@}"

echo "=== memcheck ==="
compute-sanitizer --tool=memcheck --leak-check=full $BINARY $ARGS

echo "=== racecheck ==="
compute-sanitizer --tool=racecheck $BINARY $ARGS

echo "=== synccheck ==="
compute-sanitizer --tool=synccheck $BINARY $ARGS
