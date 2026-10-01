# Repository workflow

- After completing and verifying a requested code change in this workspace, stage only the files related to that request, create a concise commit, and push the current branch to `origin`.
- Never commit generated ROS 2 directories (`build/`, `install/`, `log/`), Python caches, credentials, tokens, device logs, or other secrets.
- Preserve unrelated user changes. If a push cannot be completed, report the exact reason and leave the verified commit locally.
- Keep `src/sllidar_ros2` as the upstream Git submodule; do not vendor or modify its Git history unless the user explicitly asks.
