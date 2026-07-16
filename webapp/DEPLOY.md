# Deploying dSph posterior explorer

Two machines are involved, and it matters which one you're on for each
step:

- **Trillium** has no Docker daemon (normal for HPC login nodes -
  Docker needs root, so it's blocked cluster-wide). You can build the
  portable *bundle* here (plain files + a `Dockerfile` recipe), but
  `docker build`/`docker run` themselves cannot execute here.
- **Your laptop** (once you install Docker Desktop) is where the
  `Dockerfile` actually turns into a runnable image.

Every command below is labeled with which machine it runs on. If you
skip straight to Docker, jump to "2. On your laptop."

## 1. On Trillium: build the bundle

The bundle (`webapp/dist/dsph_explorer.tar.gz`, built by `package.py`)
is a self-contained copy of the app: the code, the trained model, and
private copies of the packages it needs, plus a `Dockerfile` and
`requirements.txt`. Nothing in it is Trillium-specific.

```bash
# on Trillium
cd ~/projects/sbi_dsph/sbi_dsph_pipeline/webapp
python package.py \
    --checkpoint-dir /scratch/tvnguyen/trained_models/npe/8p_ZhaoPlumCOM/sfaqzcwx/checkpoints \
    --checkpoint-filename last.ckpt
# -> webapp/dist/dsph_explorer.tar.gz (already built and current as of this session)
```

Download it to your laptop:

```bash
# on your laptop
scp trillium:/home/tvnguyen/projects/sbi_dsph/sbi_dsph_pipeline/webapp/dist/dsph_explorer.tar.gz .
tar xzf dsph_explorer.tar.gz && cd dsph_explorer
```

## 2. On your laptop: build and run the Docker image

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
if you haven't already - that's the only prerequisite, no Python setup
needed.

```bash
# on your laptop, inside the unpacked dsph_explorer/ directory
docker build -t dsph-explorer .
docker run -p 8799:8799 --rm dsph-explorer
```

Open `http://localhost:8799`, try `example_catalog.csv` with
`R_half = 0.2`. `--rm` cleans up the container when you stop it
(`Ctrl+C`); drop it if you'd rather leave the stopped container around.

This is the same image you'd hand to a collaborator (see §4) or deploy
to a server (§5) - building and testing it locally first is the cheap
way to catch problems before shipping it anywhere. I already did this
exact build-and-run cycle once as a validation pass (details in §3) and
found and fixed one real dependency bug in the process, so the
`Dockerfile`/`requirements.txt` you have now are known-good.

## 3. What I already validated (so you don't have to re-derive it)

Since Trillium has no Docker daemon, I validated the bundle using
**Apptainer** (available on Trillium) with a recipe that runs the
`Dockerfile`'s exact `RUN`/`COPY`/`CMD` steps - a genuine from-scratch
`pip install` into a fresh Python 3.12 environment, not a reuse of
this login node's existing packages, so it's a faithful stand-in for
what `docker build` on your laptop will do.

That from-scratch build **failed the first time**:
`requirements.txt` had `torch-geometric>=2.5` with no upper bound, so
a clean `pip install` resolved to the newest release (2.8.0), which
changed `knn_graph`'s fallback behavior and started requiring
`pyg-lib` - every inference run crashed with `ImportError:
'knn_graph' requires 'pyg-lib>=0.6.0'`. Fixed by pinning
`torch-geometric==2.7.0` in `package.py` (the version this model was
actually trained/validated against). A second from-scratch build,
followed by a full browser-driven run (upload a catalog, run
inference, confirm the interactive plots), succeeded in ~13 seconds.

This means: the `dist/dsph_explorer.tar.gz` you download in §1 already
has the fix baked in - `docker build` on your laptop should just work.
If you ever hand-edit `requirements.txt` again, re-validate with a
genuinely clean install (a fresh venv or `docker build --no-cache`,
not this login node's ambient packages) before shipping it, since
that's exactly the class of bug that only shows up on someone else's
machine.

**Bonus artifact:** that Apptainer validation also produced a working
Apptainer/Singularity image, `webapp/dist/dsph_explorer.sif`
(~414 MB, still on Trillium). If a collaborator is themselves on an
HPC cluster or other Linux box with Apptainer/Singularity (common in
astro) rather than a laptop with Docker Desktop, you can hand them
this file directly - no build step on their end at all:

```bash
# on their machine, no Docker or build tools needed
apptainer run dsph_explorer.sif   # binds 0.0.0.0:8799 directly, no -p mapping needed
```

The recipe that produced it, `webapp/dist/dsph_explorer.def`, is
saved alongside it - regenerate with `apptainer build --fakeroot
dsph_explorer.sif dsph_explorer.def` (run from the bundle's parent
directory) any time you rebuild the bundle for a new model.

## 4. On your laptop: share the image with a collaborator

If you don't need a public URL - just want a collaborator to run this
on their own machine - skip hosting entirely (§5) and hand them the
image instead. Build it once (§2), then either push it or export it
as a file:

**Option A - a registry (Docker Hub or similar), if you're OK making
it public (or they have access to a private one):**

```bash
# on your laptop
docker tag dsph-explorer yourdockerhubuser/dsph-explorer:latest
docker push yourdockerhubuser/dsph-explorer:latest
```

They then just run `docker run -p 8799:8799 yourdockerhubuser/dsph-explorer` -
Docker pulls it for them, no file transfer needed.

**Option B - a plain file, no registry/account needed:**

```bash
# on your laptop
docker save dsph-explorer | gzip > dsph-explorer.tar.gz    # ~1-2 GB
```

Send `dsph-explorer.tar.gz` however you'd send any large file (scp, a
shared drive, WeTransfer, ...). They load it with `docker load <
dsph-explorer.tar.gz`, then `docker run -p 8799:8799 dsph-explorer` as
in §2.

Either way, that's the entire recipient-side workflow: install Docker
Desktop, one `docker run`, open a browser tab - see
`webapp/RUNNING_THE_APP.md`, which is written for them directly and
covers both options plus troubleshooting.

The one downside versus real hosting (§5): it only runs while their
machine is on, and it's CPU-only (the bundled `Dockerfile` doesn't
pull in CUDA), so a run takes roughly a minute or two rather than
seconds.

## 5. Hosting it at a URL instead

If you *do* want a persistent URL rather than something collaborators
run locally, the image from §2 needs to live on a server that keeps a
process running - **not** Trillium, and **not** static hosting
(GitHub Pages, S3/Netlify-style buckets). A 1-2 CPU / 2 GB RAM box is
enough.

```bash
# on the target server (a VPS, lab machine, etc. - install Docker there too)
docker build -t dsph-explorer .        # or `docker pull` if you pushed it (§4A)
docker run -d --restart=on-failure -p 127.0.0.1:8799:8799 dsph-explorer
```

Binding to `127.0.0.1` keeps it off the open internet; put a reverse
proxy in front for the actual public URL:

```nginx
# nginx, on the same server
location /dsph/ {
    proxy_pass http://127.0.0.1:8799/;
    proxy_read_timeout 600s;   # runs can take minutes on CPU
    client_max_body_size 50m;  # catalog uploads
}
```

Add HTTPS with `certbot --nginx`.

### Access control

The app has no login of its own. Anyone who can reach the URL can
submit compute jobs on your server — and the catalog "Row filter" box
hands them a pandas `query` expression evaluated server-side. That
parser only accepts a restricted expression grammar (no imports, calls,
or attribute access), so it is not a shell, but it is still untrusted
input evaluated on your box: treat reaching the page as equivalent to
being allowed to run analysis jobs on it. If the page is public, gate
it at the proxy:

```nginx
location /dsph/ {
    ...
    auth_basic "dsph explorer";
    auth_basic_user_file /etc/nginx/.htpasswd;
}
```

```bash
sudo htpasswd -c /etc/nginx/.htpasswd yourname
```

Or just keep the URL unlisted if it's only for a handful of
collaborators.

### Where to host it

Options, roughly cheapest/simplest to most managed - all of these are
places Docker *can* run, unlike Trillium:

- **Cheap VPS** (Hetzner, DigitalOcean, Linode, ~$5-10/mo for 1-2
  CPU/2-4 GB) - you own the box, `docker run` + nginx as above. Best
  fit here: this app is stateless, low-traffic, and fine on CPU.
- **PaaS with a Dockerfile** (Fly.io, Railway, Render) - push the
  bundle's `Dockerfile`, get a public URL and HTTPS for free, no
  nginx/systemd to hand-manage. Slightly more expensive than a raw
  VPS but much less ops work.
- **Lab/university hardware behind a tunnel** (Tailscale, Cloudflare
  Tunnel, or plain SSH `-R`) - no public IP or firewall changes
  needed; good if the audience is a handful of known collaborators
  and the box already exists. Keep the container bound to
  `127.0.0.1` and let the tunnel do the exposure.
- **GPU inference platforms** (Modal, RunPod, Lambda) - overkill for
  this specific model (fine on CPU in ~1-2 min/run), but the natural
  next step if you later want sub-second responses or much larger
  posterior draws.

For this project a small VPS or a PaaS with the bundled `Dockerfile`
are the two simplest paths.

### Running without Docker at all

Docker just wraps the same install `package.py` documents - if a
target server can't or shouldn't run Docker, the non-Docker path
works too (systemd + a venv):

```bash
# on the target server
tar xzf dsph_explorer.tar.gz && sudo mv dsph_explorer /opt/ && cd /opt/dsph_explorer
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.9.0+cpu.html
pip install -r requirements.txt
python app.py   # http://<server>:8799
```

```ini
# /etc/systemd/system/dsph-explorer.service
[Unit]
Description=dSph posterior explorer
After=network.target

[Service]
WorkingDirectory=/opt/dsph_explorer
ExecStart=/opt/dsph_explorer/.venv/bin/python app.py \
    --host 127.0.0.1 --port 8799 --output-dir /opt/dsph_explorer/runs
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now dsph-explorer
```

Same nginx/access-control setup as above in front of it.

## Swapping in a different model

Replace the two files in `model/` (any Lightning `.ckpt` plus the
`config_snapshot.json` written next to it by `npe/train_npe.py`) and
rebuild - `python package.py --checkpoint-dir <other run>` on
Trillium, then repeat from §2 on whichever machine builds/runs the
image.
