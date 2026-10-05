# Release procedure

Release artifacts are published on GitHub Releases. The repository does not configure PyPI publishing.
Install a tagged release with `python -m pip install "git+https://github.com/IRVLUTD/manotorch.git@v0.1.0"`,
or download its wheel from the release page and install it with pip.

1. Update the version in `pyproject.toml`, run `uv lock`, and move the corresponding changes into a dated
   CHANGELOG entry. Review the listed behavior changes and removed APIs before upgrading downstream projects.
2. Run `uv run ruff check .`, `uv lock --check`, and `uv run pytest --require-mano` locally with both licensed
   MANO models and a CUDA device. Validate the minimum supported PyTorch environment separately. Keep model
   files, training poses, local benchmark outputs and downstream datasets outside version control.
   The development branch may retain its acceptance report. Exclude that report from master and release tags;
   release code and documentation must be in English. CI enforces this boundary and checks other source
   documents for CJK text.
3. Commit on `optimize`, push, and wait for its public CPU CI. Merge into `master` and push; wait for that
   commit's CI to pass before creating an annotated version tag.
4. Push the matching `v<version>` tag. The Release workflow repeats public CPU tests and lint, verifies that
   the tag matches the package version and is contained in `master`, builds a wheel and source distribution,
   checks package metadata and bundled assets, and publishes both files to GitHub Releases.
5. Verify the Release workflow and published artifact downloads. Record the commit, tag, CI URLs and local
   licensed-model validation results in the acceptance report. Do not move an already published tag.

Public CI runs Python 3.10 / PyTorch 2.0.1 / NumPy 1.x and Python 3.12 / PyTorch 2.11, plus Ruff.
It also verifies that an installed wheel finds its bundled anchors outside the source tree. It omits licensed
MANO models and CUDA; passing public CI does not replace the local full-model checks.

Git pushes need repository write permission. Release publishing uses the workflow's temporary `GITHUB_TOKEN`
with `contents: write` on the publishing job; it does not require a personal publishing token on the workstation.
