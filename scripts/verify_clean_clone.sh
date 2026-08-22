#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 REPOSITORY_URL NEW_DESTINATION [GIT_REF]" >&2
  exit 2
fi

repository_url="$1"
new_destination="$2"
git_ref="${3:-}"

if [[ -e "$new_destination" ]]; then
  echo "destination already exists: $new_destination" >&2
  exit 2
fi

git clone --no-local "$repository_url" "$new_destination"
cd "$new_destination"
if [[ -n "$git_ref" ]]; then
  git checkout --detach "$git_ref"
fi

./scripts/reproduce_table1.sh

attestation_args=()
if [[ "${CYTOFAIR_VIRGIN_HOST_ATTESTED:-}" == "YES" ]]; then
  attestation_args+=(--virgin-host-attested)
fi
uv run python scripts/record_clean_clone.py "${attestation_args[@]}"
