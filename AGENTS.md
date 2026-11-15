# Notes for agents

- **Before pushing code changes, retrain the PGO profile**: `sh scripts/pgo-profile.sh` (Git Bash,
  ~10 min, needs the NVIDIA GPU), then commit `pgo/`. The Windows x64 release builds use that
  committed profile; the other targets train their own in CI (`scripts/build-pgo.sh`).
- Releases: push a tag `v<version>` matching `Cargo.toml` (`.github/workflows/build.yml`).
