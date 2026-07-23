# Running the dSph posterior explorer

This is a small tool for running density-profile inference on your own
kinematic catalog - upload your data, get a posterior corner plot,
Jeans profiles, and a Wolf-mass check back. It runs entirely on your
own computer; nothing you upload leaves your machine.

You don't need Python, or any of the astro packages this tool depends
on - just [Docker](https://www.docker.com/products/docker-desktop/),
which packages the whole thing up so it runs the same way everywhere.

## 1. Install Docker (one-time)

Download **Docker Desktop** for your OS and install it like any other
app: <https://www.docker.com/products/docker-desktop/>

- Mac: works on both Intel and Apple Silicon.
- Windows: the installer will prompt you to enable WSL2 if you don't
  have it yet - just follow its instructions.
- Linux: install `docker` via your package manager, or follow Docker's
  site instructions.

Open Docker Desktop once after installing and make sure it says it's
running (there's usually a whale icon in your menu bar / system tray).
You need it running in the background any time you use the app below.

## 2. Get the image

I sent you a file called **`dsph-explorer.tar.gz`** (~1-2 GB) - save
it anywhere on your machine, e.g. your Downloads folder. Then open a
terminal in that folder and run:

```bash
docker load < dsph-explorer.tar.gz
```

This can take a minute or two (it's unpacking a couple GB). You only
need to do this once - not every time you run the app.

## 3. Run it

Open a terminal and run:

```bash
docker run -p 8799:8799 --rm dsph-explorer
```

Leave this terminal window open - it's running the app. You'll see
some log lines, ending with something like:

```
[Server] http://0.0.0.0:8799
```

That means it's ready.

## 4. Open it in your browser

Go to **<http://localhost:8799>**

You should see the "dSph posterior explorer" page.

## 5. Using it

1. **Upload your catalog** - a CSV (or ECSV/FITS table) with one row
   per star. It needs columns for RA, Dec, line-of-sight velocity and
   its error, and either a distance or a distance modulus - the app
   auto-detects these by column name, but you can fix any that don't
   match up right after uploading.
2. **Fill in the system fields** - the half-light radius (`R_half`) is
   the only one that's strictly required; everything else has a
   sensible fallback (e.g. computed from your data) if you leave it
   blank.
3. Click **Run inference**. A run with the default settings takes
   roughly 1-2 minutes on a normal laptop (this image runs on CPU).
4. You'll get an interactive profile plot (drag to zoom, hover for
   values, double-click to reset, camera icon to save a PNG) plus a
   posterior corner plot, and a link to download the raw posterior
   samples as a CSV.

I've also attached **`example_catalog.csv`** if you want to try the
app out first before using your own data - use `R_half = 0.2` with
it.

## Stopping / restarting

- To stop the app, go back to the terminal it's running in and press
  `Ctrl+C`.
- To run it again later, repeat step 3 - no need to redo the install
  or the `docker load` step.

## Troubleshooting

- **`docker: command not found`** - Docker Desktop isn't installed, or
  isn't on your PATH yet. Reopen your terminal after installing, or
  restart your computer once.
- **Docker Desktop is installed but the `docker run` command hangs or
  errors immediately** - make sure Docker Desktop is actually open and
  running (check for the whale icon), not just installed.
- **"port is already allocated" / "address already in use"** -
  something else on your machine is already using port 8799. Run it on
  a different port instead, e.g. `docker run -p 8888:8799 --rm
  dsph-explorer`, then open `http://localhost:8888` instead.
- **The first `docker run` is slow** - it's downloading/unpacking the
  image; later runs start in a few seconds.
- **A run seems stuck for a long time** - this image runs on CPU, so a
  run can take a couple of minutes, especially with a large number of
  posterior/profile samples. If it's been more than ~5 minutes, check
  the terminal for an error message.
- Nothing you upload is sent anywhere - the app and everything it
  processes stay on your own machine the whole time.

If something doesn't work, send along what the terminal printed and
I'll take a look.

## Citing this

If you use results from this tool in published work, please cite:

- Nguyen et al. (2026), "Dark Matter in Draco and Boötes I: Hints of a
  Core in an Ultra-Faint Dwarf from Simulation-Based Inference",
  arXiv:2606.26218 — <https://doi.org/10.48550/arXiv.2606.26218>
- Nguyen et al. (2025), "Trial by FIRE: probing the dark matter density
  profile of dwarf galaxies with GraphNPE", MNRAS 541, 2707 —
  <https://doi.org/10.1093/mnras/staf1118>
- Nguyen et al. (2023), "Uncovering dark matter density profiles in
  dwarf galaxies with graph neural networks", Phys. Rev. D 107, 043015 —
  <https://doi.org/10.1103/PhysRevD.107.043015>

The same list is in the app's page footer.
