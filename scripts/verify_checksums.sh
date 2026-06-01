#!/bin/bash
#
# Verify checksums of downloaded Spotless benchmark archives.
#
# Two checksum files are provided:
#   checksums/SHA256SUMS  — independently computed SHA-256 hashes (primary)
#   checksums/MD5SUMS     — MD5 hashes from the Zenodo API (fallback)
#
# Usage:
#     bash scripts/verify_checksums.sh [DATA_DIR]
#
# Example:
#     bash scripts/verify_checksums.sh ./data/spotless

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${1:-./data/spotless}"

# Prefer SHA-256; fall back to MD5
SHA256_FILE="$REPO_ROOT/checksums/SHA256SUMS"
MD5_FILE="$REPO_ROOT/checksums/MD5SUMS"

if [ -f "$SHA256_FILE" ]; then
    CHECKSUM_FILE="$SHA256_FILE"
    ALGO="SHA-256"
    if command -v sha256sum &>/dev/null; then
        hashcmd="sha256sum"
    elif command -v shasum &>/dev/null; then
        hashcmd="shasum -a 256"
    else
        echo "WARNING: No sha256sum or shasum found; falling back to MD5." >&2
        CHECKSUM_FILE="$MD5_FILE"
        ALGO="MD5"
    fi
fi

if [ "$ALGO" = "MD5" ] 2>/dev/null || [ ! -f "$SHA256_FILE" ]; then
    CHECKSUM_FILE="${CHECKSUM_FILE:-$MD5_FILE}"
    ALGO="MD5"
    if command -v md5sum &>/dev/null; then
        hashcmd="md5sum"
    elif command -v md5 &>/dev/null; then
        hashcmd="md5 -r"
    else
        echo "ERROR: No checksum command found (tried sha256sum, shasum, md5sum, md5)." >&2
        exit 1
    fi
fi

if [ ! -f "$CHECKSUM_FILE" ]; then
    echo "ERROR: Checksum file not found: $CHECKSUM_FILE" >&2
    exit 1
fi

echo "============================================================"
echo "Verifying archive checksums ($ALGO)"
echo "Data directory: $DATA_DIR"
echo "Checksum file:  $CHECKSUM_FILE"
echo "============================================================"

ERRORS=0
CHECKED=0

# Read each checksum line (skip comments and blank lines)
while IFS= read -r line; do
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

    expected_hash=$(echo "$line" | awk '{print $1}')
    filename=$(echo "$line" | awk '{print $2}')
    filepath="$DATA_DIR/$filename"

    if [ ! -f "$filepath" ]; then
        echo "  SKIP  $filename (not yet downloaded)"
        continue
    fi

    echo -n "  Checking $filename ($ALGO) ... "
    actual_hash=$($hashcmd "$filepath" | awk '{print $1}')

    if [ "$actual_hash" = "$expected_hash" ]; then
        echo "OK"
        CHECKED=$((CHECKED + 1))
    else
        echo "FAILED"
        echo "    Expected: $expected_hash" >&2
        echo "    Got:      $actual_hash" >&2
        ERRORS=$((ERRORS + 1))
    fi
done < "$CHECKSUM_FILE"

echo ""
if [ "$ERRORS" -gt 0 ]; then
    echo "ERROR: $ERRORS file(s) failed checksum verification." >&2
    echo "  Delete the corrupted files and re-download:" >&2
    echo "    bash scripts/download_spotless_data.sh $DATA_DIR" >&2
    exit 1
elif [ "$CHECKED" -eq 0 ]; then
    echo "No archives found to verify (files not yet downloaded)."
else
    echo "All $CHECKED archive(s) verified ($ALGO)."
fi
