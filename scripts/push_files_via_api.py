"""Push local changes to GitHub through api.github.com (single commit).

Why
---
On this machine ``github.com:443`` is being reset (``errno 10054`` /
``Operation timed out``), so ``git push`` cannot reach the remote. The API host
``api.github.com`` is reachable, and the Git Data API can build a commit exactly
like git would: create blobs, build a tree on top of the current one, create the
commit, move the branch ref. Result is **one** normal commit on ``main``.

Usage
-----
    export GITHUB_TOKEN=ghp_xxx      (Git Bash)
    set GITHUB_TOKEN=ghp_xxx         (cmd)
    $env:GITHUB_TOKEN="ghp_xxx"      (PowerShell)

    python scripts/push_files_via_api.py            # dry run
    python scripts/push_files_via_api.py --apply    # commit for real

The token needs the ``repo`` scope (classic) or "Contents: write" (fine-grained).
Nothing is sent anywhere except api.github.com.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "1347993373/INDNet-LLIE-Detection"
BRANCH = "main"
API = "https://api.github.com"

FILES = [
    "LICENSE",
    "README.md",
    "scripts/pack_weights.py",
    "docs/weights_README.txt",
    "scripts/push_files_via_api.py",
]
MESSAGE = "Add author and Zenodo DOI; add weight-bundling helper"


def call(method: str, url: str, token: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "indnet-release-script")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")[:400]}
    except urllib.error.URLError as e:
        return 0, {"error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--files", nargs="*", default=FILES)
    ap.add_argument("--message", default=MESSAGE)
    a = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    # 常见污染：首尾引号、空格、换行、Windows 粘贴带的 \r
    token = token.strip().strip('"').strip("'").strip()

    if token:
        kind = ("classic" if token.startswith("ghp_")
                else "fine-grained" if token.startswith("github_pat_")
                else "unknown")
        print(f"token: {token[:4]}...  length={len(token)}  ({kind})")
        if kind == "unknown":
            print("  [warn] 前缀既不是 ghp_ 也不是 github_pat_ —— 多半粘错了内容")
        if len(token) < 40:
            print("  [warn] 长度不足 40 —— 很有可能没复制完整（或者复制的是 token 名称）")
        if any(c.isspace() for c in token):
            print("  [warn] token 里含有空白字符 —— 中间混进了空格/换行")

    if a.apply and not token:
        raise SystemExit("GITHUB_TOKEN not set -- see the docstring.")

    st, me = call("GET", f"{API}/user", token)
    if token and st != 200:
        hint = ""
        if st == 401:
            hint = ("\n  token 无效。按顺序查：\n"
                    "   1) echo ${#GITHUB_TOKEN}         长度应为 40（classic）/ 90+（fine-grained）\n"
                    "   2) echo ${GITHUB_TOKEN:0:4}      应为 ghp_ 或 gith\n"
                    "   3) 是否在别的终端窗口里 export 的（环境变量不跨窗口）\n"
                    "   4) token 是否已过期、或曾经暴露被 GitHub 自动吊销\n"
                    "   5) fine-grained token 必须在 'Repository access' 里勾上 INDNet-LLIE-Detection\n")
        elif st == 403:
            hint = "\n  token 有效但权限不够：classic 需要 repo，fine-grained 需要 Contents: write。\n"
        raise SystemExit(f"token rejected ({st}): {me.get('error', me)}{hint}")
    if token:
        print(f"authenticated as {me.get('login')}\n")

    st, ref = call("GET", f"{API}/repos/{REPO}/git/ref/heads/{BRANCH}", token)
    if st != 200:
        raise SystemExit(f"cannot read ref ({st}): {ref.get('error', ref)}")
    head_sha = ref["object"]["sha"]

    st, commit = call("GET", f"{API}/repos/{REPO}/git/commits/{head_sha}", token)
    if st != 200:
        raise SystemExit(f"cannot read head commit ({st}): {commit.get('error', commit)}")
    base_tree = commit["tree"]["sha"]
    print(f"branch {BRANCH} at {head_sha[:7]}  tree {base_tree[:7]}\n")

    blobs, listing = [], []
    for rel in a.files:
        local = ROOT / rel
        if not local.is_file():
            print(f"  [skip] {rel}: missing locally")
            continue
        raw = local.read_bytes()
        listing.append((rel, len(raw)))
        blobs.append((rel, base64.b64encode(raw).decode()))

    if not blobs:
        raise SystemExit("nothing to upload")

    for rel, n in listing:
        print(f"  [file] {rel:<34} {n:>7} bytes")

    if not a.apply:
        print(f"\n{len(blobs)} file(s) ready. Dry run -- add --apply to commit.")
        return 0

    tree = []
    for rel, b64 in blobs:
        st, out = call("POST", f"{API}/repos/{REPO}/git/blobs", token,
                       {"content": b64, "encoding": "base64"})
        if st not in (200, 201):
            raise SystemExit(f"blob failed for {rel}: {st} {out.get('error', out)}")
        tree.append({"path": rel, "mode": "100644", "type": "blob", "sha": out["sha"]})
        print(f"  [blob] {rel} -> {out['sha'][:7]}")

    st, out = call("POST", f"{API}/repos/{REPO}/git/trees", token,
                   {"base_tree": base_tree, "tree": tree})
    if st not in (200, 201):
        raise SystemExit(f"tree failed: {st} {out.get('error', out)}")
    new_tree = out["sha"]
    print(f"  [tree] {new_tree[:7]}")

    st, out = call("POST", f"{API}/repos/{REPO}/git/commits", token,
                   {"message": a.message, "tree": new_tree, "parents": [head_sha]})
    if st not in (200, 201):
        raise SystemExit(f"commit failed: {st} {out.get('error', out)}")
    new_commit = out["sha"]
    print(f"  [commit] {new_commit[:7]}  {a.message}")

    st, out = call("PATCH", f"{API}/repos/{REPO}/git/refs/heads/{BRANCH}", token,
                   {"sha": new_commit, "force": False})
    if st not in (200, 201):
        raise SystemExit(f"ref update failed: {st} {out.get('error', out)}")

    print(f"\nbranch {BRANCH} -> {new_commit[:7]}")
    print("Next:  git fetch origin && git reset --soft origin/main")
    print("       (local tree is identical to the remote one, so nothing is lost)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
