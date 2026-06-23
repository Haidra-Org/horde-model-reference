"""Maintainer-facing tooling to mirror hostable (non-generation) model files onto the gated Cloudflare R2 bucket.

See :mod:`scripts.r2_sync.sync` for the CLI entry point and the module docstrings for the design. The tool is
content-addressed (``by-hash/<sha256>``), opt-in by a redistributable allowlist, idempotent, and backfills the
real sha256 of records that still carry the ``"FIXME"`` sentinel.
"""
