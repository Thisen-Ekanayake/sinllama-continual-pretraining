#!/usr/bin/env bash
# Shared helper: sync a local results directory to a GCS destination.
# Source this from a task's run_*.sh, then call:
#   upload_to_gcs "$OUT_DIR" "$BUCKET_DEST"
#
# No-op (with a warning) if gsutil isn't installed or BUCKET_DEST is empty —
# callers should not have to guard every invocation themselves.

upload_to_gcs() {
  local local_dir="$1" bucket_dest="$2"
  if [[ -z "$bucket_dest" ]]; then
    echo "[$(date '+%F %T')] (no bucket configured — skipping upload of $local_dir)"
    return 0
  fi
  if ! command -v gsutil >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] WARNING: gsutil not found — skipping upload of $local_dir -> $bucket_dest"
    return 0
  fi
  echo "[$(date '+%F %T')] Uploading $local_dir -> $bucket_dest"
  gsutil -m rsync -r "$local_dir" "$bucket_dest" \
    || echo "[$(date '+%F %T')] WARNING: gsutil rsync failed for $local_dir -> $bucket_dest"
}
