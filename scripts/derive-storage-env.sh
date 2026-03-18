#!/usr/bin/env sh
set -eu

# Derive full APP_STORAGE_S3_* contract variables from minimal inputs.
#
# Minimal required inputs:
# - APP_STORAGE_S3_URL (full URL, optionally including /bucket)
# - APP_STORAGE_S3_ACCESS_KEY_ID
# - APP_STORAGE_S3_SECRET_ACCESS_KEY
#
# Usage:
#   eval "$(/app/scripts/derive-storage-env.sh --export)"

output_mode="plain"
if [ "${1:-}" = "--export" ]; then
  output_mode="export"
fi

emit() {
  key="$1"
  value="$2"
  if [ "$output_mode" = "export" ]; then
    printf 'export %s=%s\n' "$key" "$(printf '%s' "$value" | sed "s/'/'\\''/g; s/^/'/; s/$/'/")"
  else
    printf '%s=%s\n' "$key" "$value"
  fi
}

require() {
  key="$1"
  eval "val=\${$key:-}"
  if [ -z "$val" ]; then
    echo "missing required env: $key" >&2
    exit 1
  fi
}

default_if_empty() {
  key="$1"
  value="$2"
  eval "current=\${$key:-}"
  if [ -z "$current" ]; then
    eval "$key=\"$value\""
  fi
}

default_if_empty APP_STORAGE_MODE "s3"
default_if_empty APP_STORAGE_ROOT "/workspace"
default_if_empty APP_STORAGE_S3_REGION "auto"
default_if_empty APP_STORAGE_S3_PATH_STYLE "true"

require APP_STORAGE_S3_ACCESS_KEY_ID
require APP_STORAGE_S3_SECRET_ACCESS_KEY

if [ -z "${APP_STORAGE_S3_ENDPOINT:-}" ] || [ -z "${APP_STORAGE_S3_BUCKET:-}" ]; then
  require APP_STORAGE_S3_URL

  url="${APP_STORAGE_S3_URL%/}"
  url_no_query="${url%%\?*}"
  url_no_fragment="${url_no_query%%#*}"

  case "$url_no_fragment" in
    *://*)
      scheme="${url_no_fragment%%://*}"
      remainder="${url_no_fragment#*://}"
      ;;
    *)
      scheme="https"
      remainder="$url_no_fragment"
      ;;
  esac

  host="${remainder%%/*}"
  if [ "$host" = "$remainder" ]; then
    path=""
  else
    path="${remainder#*/}"
  fi

  if [ -z "${APP_STORAGE_S3_ENDPOINT:-}" ]; then
    APP_STORAGE_S3_ENDPOINT="$scheme://$host"
  fi

  if [ -z "${APP_STORAGE_S3_BUCKET:-}" ] && [ -n "$path" ]; then
    APP_STORAGE_S3_BUCKET="${path%%/*}"
  fi
fi

require APP_STORAGE_S3_ENDPOINT
require APP_STORAGE_S3_BUCKET

if [ -z "${APP_STORAGE_S3_PREFIX:-}" ]; then
  session_hint="${APP_SESSION_ID:-manual}"
  APP_STORAGE_S3_PREFIX="sessions/$session_hint"
fi

emit APP_STORAGE_MODE "$APP_STORAGE_MODE"
emit APP_STORAGE_ROOT "$APP_STORAGE_ROOT"
emit APP_STORAGE_S3_ACCESS_KEY_ID "$APP_STORAGE_S3_ACCESS_KEY_ID"
emit APP_STORAGE_S3_SECRET_ACCESS_KEY "$APP_STORAGE_S3_SECRET_ACCESS_KEY"
emit APP_STORAGE_S3_ENDPOINT "$APP_STORAGE_S3_ENDPOINT"
emit APP_STORAGE_S3_BUCKET "$APP_STORAGE_S3_BUCKET"
emit APP_STORAGE_S3_REGION "$APP_STORAGE_S3_REGION"
emit APP_STORAGE_S3_PREFIX "$APP_STORAGE_S3_PREFIX"
emit APP_STORAGE_S3_PATH_STYLE "$APP_STORAGE_S3_PATH_STYLE"

if [ -n "${APP_STORAGE_S3_SESSION_TOKEN:-}" ]; then
  emit APP_STORAGE_S3_SESSION_TOKEN "$APP_STORAGE_S3_SESSION_TOKEN"
fi
