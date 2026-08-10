# phantomkit Docker Image

A self-contained image for running the phantomkit MRI phantom QA pipeline, bundling all required neuroimaging tools.

## What's included

| Tool | Version | Purpose |
|------|---------|---------|
| Ubuntu | 24.04 | Base OS |
| Python | 3.12 | Runtime |
| MRtrix3 | system apt | DWI preprocessing, format conversion |
| FSL | latest (conda installer) | `eddy`, `topup`, `flirt` |
| ANTs | 2.5.3 | Registration (`antsRegistrationSyN.sh`, `antsApplyTransforms`) |
| dcm2niix | latest | DICOM → NIfTI conversion |
| phantomkit | repo source | This package |

> **Note:** The FSL layer alone is ~7 GB. Expect a final image size of ~10–12 GB and a build time of 30–60 minutes depending on network speed.

---

## Pulling on another machine

```bash
docker pull arkiev/phantomkit:latest
```
---


## Running the container

### Full pipeline

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v /path/to/input:/data/input \
  -v /path/to/output:/data/output \
  phantomkit:latest \
  pipeline \
    --input-dir /data/input \
    --output-dir /data/output \
    --phantom SPIRIT
```

`--user "$(id -u):$(id -g)"` makes the container write files as *you* instead
of root, so the output directory doesn't end up requiring `sudo` to modify
afterwards (see [Troubleshooting](#troubleshooting)).

### Plotting (compare two HTML reports)

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v /path/to/plots:/data/plots \
  phantomkit:latest \
  plot compare-plots \
    /data/plots/T1_mapping.html \
    /data/plots/T1_mapping_reduced.html \
    -o /data/plots/comparison.html
```

### GUI

**Easiest option — no typing needed after the first setup:** download
[`phantomkit-gui.sh`](phantomkit-gui.sh) (macOS/Linux) or
[`phantomkit-gui.bat`](phantomkit-gui.bat) (Windows), make sure Docker Desktop
is running, then double-click the file. It starts the container, mounts your
whole home/user folder so the GUI's file browser can reach your data, waits
for the server to come up, and opens it in your default browser
automatically. Run it again any time to reopen the GUI; it reuses the
already-running container if one exists.

To stop it: `docker stop phantomkit-gui`.

**Manual equivalent**, if you'd rather run the command yourself:

```bash
docker run -d --rm \
  --name phantomkit-gui \
  --user "$(id -u):$(id -g)" \
  -p 7878:7878 \
  -v "$HOME:/hostuser" \
  -e PHANTOMKIT_HOME=/hostuser \
  -e HOME=/tmp \
  phantomkit:latest gui
```

Then open `http://localhost:7878`. `PHANTOMKIT_HOME` tells the GUI's folder
browser where to start — point it at whatever host directory you mount so
you're not stuck browsing the container's empty internal filesystem.
`--user "$(id -u):$(id -g)"` (with `HOME` overridden to a scratch dir the
container can actually write to) makes outputs come out owned by you instead
of root — see [Troubleshooting](#troubleshooting).

### Interactive shell (debugging)

```bash
docker run --rm -it \
  -v /path/to/input:/data/input \
  -v /path/to/output:/data/output \
  --entrypoint bash \
  phantomkit:latest
```

### List available commands

```bash
docker run --rm phantomkit:latest --help
```

## Building and pushing the image

The image targets both `linux/amd64` (Linux servers) and `linux/arm64` (Apple Silicon). Use `buildx` to build and push in one step.

```bash
docker login
docker buildx create --use
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -f docker/Dockerfile \
  -t arkiev/phantomkit:latest \
  -t arkiev/phantomkit:0.1.8 \
  --push .
```

For a local-only build (single platform, no push):

```bash
docker build -f docker/Dockerfile -t arkiev/phantomkit:latest .
```

---


---

## Volume mounts

| Container path | Purpose |
|----------------|---------|
| `/data/input`  | Input data (DICOM, NIfTI, or MIF) |
| `/data/output` | Processing outputs |

Any additional paths can be mounted with `-v`.

---


---

## Troubleshooting

**`template_data` not found:** The image uses an editable install so that `template_data/` is resolved relative to `/opt/phantomkit/`. If you override `WORKDIR` or move files, this may break.

**ANTs not found on PATH:** Confirm the symlink exists: `docker run --rm --entrypoint which phantomkit:latest antsRegistration`

**FSL eddy not found:** Verify `FSLDIR`: `docker run --rm --entrypoint bash phantomkit:latest -c 'echo $FSLDIR && ls $FSLDIR/bin/eddy*'`

**Output files owned by root / need `sudo` to edit or delete them:** The
container has no `USER` set, so by default it runs as root — on Linux (and
some Docker Desktop configurations) that means anything it writes into a
bind-mounted output folder is owned by root on the host too. Fix it by
passing `--user "$(id -u):$(id -g)"` on `docker run` (as in the examples
above), which makes the container write as your own user/group instead. If
you already have root-owned files from a previous run, reclaim them once
with `sudo chown -R "$(id -u):$(id -g)" /path/to/output`. The GUI launcher
scripts ([phantomkit-gui.sh](phantomkit-gui.sh) /
[phantomkit-gui.bat](phantomkit-gui.bat)) already do this for you.
