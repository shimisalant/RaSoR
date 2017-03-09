#!/bin/bash

# This script compiles and packages the Java source code.

set -e
set -o pipefail

DIR=$(dirname "$0")
TMP_DIR=$(mktemp -d -p "$DIR")
JAVA_FILE="$DIR/SquadTokenizer.java"
JAVA_CP="$DIR/*"
JAR_FILE="$DIR/squad-tokenizer.jar"

echo "Compiling $JAVA_FILE"
javac -cp "$JAVA_CP" -d "$TMP_DIR" "$JAVA_FILE"
echo "Packaging $JAR_FILE"
jar cf "$JAR_FILE" -C "$TMP_DIR" .
rm -rf "$TMP_DIR"

