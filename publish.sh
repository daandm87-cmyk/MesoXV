#!/usr/bin/env bash
# Push the site folder ($OUT) to the gh-pages branch as ONE fresh commit each time
# (force push), so the branch never accumulates the history of every 15-minute map.
# Called by run.py after each frame when MESOXV_PUBLISH points here.
set -euo pipefail
cd "$OUT"
if [ ! -d .git ]; then
  git init -q
  git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
fi
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git checkout -q --orphan next
git add -A
git commit -q -m "MesoXV maps $(date -u +'%Y-%m-%d %H:%M UTC')"
git branch -D gh-pages >/dev/null 2>&1 || true
git branch -m gh-pages
for attempt in 1 2 3; do
  git push -q -f origin gh-pages && exit 0
  sleep 10
done
exit 1
