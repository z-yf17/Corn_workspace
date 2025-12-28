#!/usr/bin/env bash
set -e

NAME="foundationpose"
DIR="$(cd "$(dirname "$0")/../" && pwd)/../"   # 保持你原来的 DIR 逻辑
# ↑ 如果你确定 DIR 就是 run_container.sh 所在目录的上一级再上一级，可按需调整

# 允许 X11
xhost + >/dev/null 2>&1 || true

if docker inspect "$NAME" >/dev/null 2>&1; then
  echo "[fp] container exists: $NAME"
  if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != "true" ]]; then
    echo "[fp] starting container: $NAME"
    docker start "$NAME" >/dev/null
  fi
else
  echo "[fp] creating container: $NAME"
  docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host \
    --name "$NAME" \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$DIR:$DIR" -v /home:/home -v /mnt:/mnt \
    -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp \
    --ipc=host \
    -e DISPLAY="${DISPLAY}" -e GIT_INDEX_FILE \
    foundationpose:latest \
    bash -lc "cd \"$DIR\" && exec bash"
fi

# 进入已有容器（不会删）
echo "[fp] attach to container: $NAME"
docker exec -it "$NAME" bash -lc "cd \"$DIR\" && exec bash"

