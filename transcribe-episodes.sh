#!/usr/bin/env bash
# Wrapper — use run-batch.sh (Whisper large-v3-yue, highest accuracy on Intel Mac).
# SenseVoice via transcribe-anything does not work on macOS x86_64.
exec "$(dirname "$0")/run-batch.sh" "$@"
