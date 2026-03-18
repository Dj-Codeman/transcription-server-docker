#!/bin/sh
set -eu

mode="${APP_STORAGE_MODE:-none}"
root="${APP_STORAGE_ROOT:-/workspace}"

require() {
  key="$1"
  eval "val=\${$key:-}"
  if [ -z "$val" ]; then
    echo "missing required env: $key" >&2
    exit 1
  fi
}

case "$mode" in
  none)
    echo "storage mode: none"
    ;;
  mounted)
    mount_path="${APP_STORAGE_MOUNT_PATH:-$root}"
    if [ ! -d "$mount_path" ]; then
      echo "mount path not found: $mount_path" >&2
      exit 1
    fi
    if [ "${APP_STORAGE_READONLY:-false}" != "true" ]; then
      touch "$mount_path/.storage-write-check" || {
        echo "mount path not writable: $mount_path" >&2
        exit 1
      }
      rm -f "$mount_path/.storage-write-check"
    fi
    echo "storage mode: mounted (ok)"
    ;;
  s3)
    require APP_STORAGE_S3_ENDPOINT
    require APP_STORAGE_S3_BUCKET
    require APP_STORAGE_S3_REGION
    require APP_STORAGE_S3_PREFIX
    require APP_STORAGE_S3_ACCESS_KEY_ID
    require APP_STORAGE_S3_SECRET_ACCESS_KEY
    echo "storage mode: s3 (env ok)"
    ;;
  *)
    echo "invalid APP_STORAGE_MODE: $mode" >&2
    exit 1
    ;;
esac
