#!/usr/bin/env bash
# Fetch the external UAV-Dot code this project builds on, pinned to the commits behind every reported result, and
# create its Python 3.10 environment from requirements-uavdot.txt. Both repositories are Apache-2.0 and stay outside
# Git (external/ is ignored). The released checkpoints are downloaded by hand (README), then checked with
#   sha256sum -c scripts/uavdot_checkpoints.sha256
set -euo pipefail
cd "$(dirname "$0")/.."

clone() {  # url directory commit; the upstream submodules use SSH URLs, so rewrite them to HTTPS
    local https=(-c url."https://github.com/".insteadOf="git@github.com:")
    [ -d "$2/.git" ] || git "${https[@]}" clone -q "$1" "$2"
    git -C "$2" checkout -q "$3"
    git "${https[@]}" -C "$2" submodule update -q --init --recursive
}
clone https://github.com/up-count/uav-dot.git external/uav-dot a6d3664b50cc0cc6f1a08848eb294b71753b6600
clone https://github.com/up-count/uav-dot-track.git external/uav-dot-track 505a7289b9602524383ae4260912f8e783000ccf

[ -x external/uav-dot/.venv/bin/python ] || uv venv -q --python 3.10 external/uav-dot/.venv
uv pip install -q --python external/uav-dot/.venv/bin/python -r requirements-uavdot.txt \
    --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu121

# The tracker expects the released detector checkpoints in its own checkpoints/ folder.
mkdir -p external/uav-dot/checkpoints external/uav-dot-track/checkpoints
for checkpoint in dot_pd_dronecrowd_51.00.ckpt dot_pd_upcount_66.49.ckpt; do
    ln -sfn "../../uav-dot/checkpoints/$checkpoint" "external/uav-dot-track/checkpoints/$checkpoint"
done
echo "Done. Put the released checkpoints in external/uav-dot/checkpoints/ and verify them with:"
echo "  sha256sum -c scripts/uavdot_checkpoints.sha256"
