#!/usr/bin/env bash
# Resumable download of a public (non-gated) Hugging Face repo at a pinned commit, with size + hash checks.
#   bash scripts/download_hf.sh google/gemma-4-26B-A4B-it 4d7ae4984b7db7de8f8457170b3f1a419ee76d52 models/gemma-4-26B-A4B-it
# Needs <dest>/_manifest.txt: "<file> <size> <lfs sha256 or -> <git blob sha1>" per line (from the HF API, see README).
# Large (LFS) files are checked by SHA-256, small files by their git blob id. No token is used or stored.
# Rerun to resume; complete files are not downloaded again, only checked.
set -u
REPO="$1"; REV="$2"; DEST="$3"
cd "$DEST" || exit 1
fail=0
while read -r f size sha blob; do
  f=${f%$'\r'}; size=${size%$'\r'}; sha=${sha%$'\r'}; blob=${blob%$'\r'}   # tolerate CRLF manifests
  [ -z "$f" ] && continue
  case "$f" in .eval_results/*|.gitattributes|README.md) continue ;; esac
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$have" != "$size" ]; then
    echo "$(date '+%F %T') get $f ($have / $size bytes)"
    mkdir -p "$(dirname "$f")"
    for try in 1 2 3 4 5 6 7 8 9 10; do
      curl -sS -L --fail -C - --retry 5 --retry-delay 10 -o "$f" "https://huggingface.co/$REPO/resolve/$REV/$f" && break
      echo "$(date '+%F %T') retry $try for $f"; sleep 15
    done
  fi
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$have" != "$size" ]; then echo "SIZE MISMATCH $f: $have != $size"; fail=1; continue; fi
  if [ -n "$sha" ] && [ "$sha" != "-" ]; then
    got=$(sha256sum "$f" | cut -d' ' -f1)
    if [ "$got" = "$sha" ]; then echo "$(date '+%F %T') ok $f (sha256 match)"; else echo "SHA256 MISMATCH $f: $got"; fail=1; fi
  elif [ -n "$blob" ]; then
    got=$(git hash-object --no-filters "$f")
    if [ "$got" = "$blob" ]; then echo "$(date '+%F %T') ok $f (git blob match)"; else echo "BLOB MISMATCH $f: $got"; fail=1; fi
  else
    echo "$(date '+%F %T') ok $f ($size bytes, no hash available)"
  fi
done < _manifest.txt
[ $fail = 0 ] && echo "ALL FILES OK" || echo "SOME FILES FAILED"
exit $fail
