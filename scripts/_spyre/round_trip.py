# SPDX-License-Identifier: Apache-2.0
"""Vendored Spyre round-trip lowering — Triton ``@triton.jit`` -> KTIR.

This module is a **trimmed, self-contained vendoring** of the Spyre Triton
fork's ``third_party/spyre/scripts/dump_round_trip.py`` (plus the
``compile_to_ttir`` / ``make_ktir_mod`` helpers from ``test/utils.py`` and
``clean_ir`` from ``scripts/_patterns/__init__.py``).

Why vendored: ``scripts/gen_ktir.py`` needs the round-trip lowering entry
point, but a packaged (non-editable) ``torch-spyre/triton`` install ships only
``python/triton/`` — it drops ``third_party/spyre/{scripts,test}``. Vendoring
the ~3 helpers we actually use lets the spyre backend come from an ordinary
(non-editable) install, so the repo no longer needs a hand-managed
``external/triton`` checkout on the default path.

Upstream origin
---------------
  repo: https://github.com/fabianlim/triton   (PR #19, torch-spyre/triton#19)
  rev:  7814b672  (3.7.0+git7814b672)
  files: third_party/spyre/scripts/dump_round_trip.py  (--driver path only)
         third_party/spyre/test/utils.py                (compile_to_ttir, make_ktir_mod)
         third_party/spyre/scripts/_patterns/__init__.py (clean_ir)

The one substantive change from upstream is the SpyreBackend import: upstream
relies on ``third_party/spyre/`` being on ``sys.path`` so ``from backend.compiler
import SpyreBackend`` resolves against the source tree. A packaged install
exposes the same class at ``triton.backends.spyre.compiler``, which is what we
import here. Keep this file in sync when the upstream lowering API changes;
``scripts/gen_ktir.py --check`` is the drift guard for the *output*.

Usage (driven by gen_ktir.py; not meant to be run by hand)::

    python -m scripts._spyre.round_trip --driver kernels/<name>/lower.py --dest <dir>

writes ``<dest>/<driver-stem>/<driver-stem>.ktir``.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# clean_ir  (from _patterns/__init__.py) — strip loc(...) / #loc noise and
# add section-break blank lines so the emitted KTIR is readable.
# ---------------------------------------------------------------------------

# Match ``loc(...)`` including one level of nesting (e.g. ``loc("x"(#loc))``).
_LOC_CALL = re.compile(r"\s*loc\((?:[^()]|\([^()]*\))*\)")

# Ops that mark a new logical section; a blank line is inserted before the
# first occurrence of each in a run.
_SECTION_BREAK_OPS = (
    "tt.get_program_id",
    "tt.make_tensor_descriptor",
    "ktdp.construct_memory_view",
    "ktdp.construct_access_tile",
    "ktdp.construct_indirect_access_tile",
    "ktdp.get_compute_tile_id",
    "scf.for",
    "tt.return",
    "func.return",
)


def _match_section_op(line: str) -> str | None:
    stripped = line.lstrip()
    op_part = re.sub(r"^%\S+\s*=\s*", "", stripped)
    for op in _SECTION_BREAK_OPS:
        if op_part.startswith(op):
            return op
    return None


def _add_section_breaks(lines: list[str]) -> str:
    out: list[str] = []
    last_section_op: str | None = None
    for line in lines:
        op = _match_section_op(line)
        if op is not None and op != last_section_op:
            if out and out[-1].strip() and not out[-1].rstrip().endswith("{"):
                out.append("")
        if line.strip():
            last_section_op = op
        out.append(line)
    return "\n".join(out)


def clean_ir(text: str) -> str:
    """Strip ``loc(...)`` / ``#loc`` noise and add section-break blank lines."""
    text = _LOC_CALL.sub("", text)
    lines = [l for l in text.split("\n") if not l.strip().startswith("#loc")]
    return _add_section_breaks(lines)


# ---------------------------------------------------------------------------
# compile_to_ttir  (from test/utils.py) — @triton.jit -> TTIR text.
# ---------------------------------------------------------------------------

def compile_to_ttir(kernel_fn, signature, constexprs) -> str:
    """Compile a ``@triton.jit`` function to TTIR text.

    Parameters
    ----------
    kernel_fn  : a ``@triton.jit`` decorated function (``triton.JITFunction``)
    signature  : dict mapping arg names to type strings (e.g. ``"*fp32"``)
    constexprs : dict mapping constexpr names to values
    """
    from triton._C.libtriton import ir
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from triton.backends.spyre.compiler import SpyreBackend

    target = GPUTarget(backend="spyre", arch=1, warp_size=1)
    src = ASTSource(fn=kernel_fn, signature=signature, constexprs=constexprs)

    backend = SpyreBackend(target)
    options = backend.parse_options({})

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    codegen_fns = (backend.get_codegen_implementation(options)
                   if hasattr(backend, "get_codegen_implementation") else {})
    module_map = backend.get_module_map()

    mod = src.make_ir(target, options, codegen_fns, module_map, context)
    return str(mod)


# ---------------------------------------------------------------------------
# make_ktir_mod  (from test/utils.py) — TTIR -> KTIR pipeline, live module.
# ---------------------------------------------------------------------------

def make_ktir_mod(ttir_path, *, grid=None):
    """Parse *ttir_path*, run TTIR and KTIR passes, return the live module.

    ``grid`` is an optional per-axis hardware partition forwarded to the
    DistributeWork pass via SpyreOptions. Defaults to the backend's default
    grid (currently ``(32,)``) when omitted.
    """
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.spyre.compiler import SpyreBackend

    target = GPUTarget(backend="spyre", arch=1, warp_size=1)
    backend = SpyreBackend(target)
    opts = {"grid": tuple(grid)} if grid is not None else {}
    options = backend.parse_options(opts)

    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)

    mod = ir.parse_mlir_module(str(ttir_path), ctx)
    mod.context = ctx

    metadata: dict = {}
    mod = backend._make_ttir(mod, metadata, options)
    return backend._make_ktir(mod, metadata, options)


# ---------------------------------------------------------------------------
# _run_make_ttir  (from dump_round_trip.py) — post-inline the raw TTIR so the
# emitted distribution is one self-contained tt.func (no un-inlined tt.call).
# ---------------------------------------------------------------------------

def _run_make_ttir(ttir_text: str):
    """Parse raw TTIR text, run ``SpyreBackend._make_ttir`` (inliner +
    canonicalizer + combine + reorder_broadcast + CSE + symbol_dce), return
    ``(mod, text)``.

    ``compile_to_ttir`` returns the raw output of ``ASTSource.make_ir`` —
    kernels that call ``tl.*`` helpers still have un-inlined ``tt.call`` and the
    callee ``tt.func private @triton.language...`` definitions. The distribution
    output should be post-inlining so reviewers see one self-contained
    ``tt.func``.
    """
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.spyre.compiler import SpyreBackend

    target = GPUTarget(backend="spyre", arch=1, warp_size=1)
    backend = SpyreBackend(target)
    options = backend.parse_options({})

    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete_on_close=False) as f:
        f.write(ttir_text)
        f.flush()
        mod = ir.parse_mlir_module(f.name, ctx)
    mod.context = ctx
    mod = backend._make_ttir(mod, {}, options)
    return mod, str(mod)


def compile_variant(entry: dict) -> tuple[str, str]:
    """Compile one driver entry to ``(ttir_text, ktir_text)``, both cleaned
    and post-inlining."""
    grid = entry.get("grid")  # None -> backend default

    raw_ttir = compile_to_ttir(
        entry["kernel_fn"],
        entry["signature"],
        entry.get("constexprs", {}),
    )
    _, ttir_text = _run_make_ttir(raw_ttir)

    # KTIR: feed the post-inlined TTIR back through make_ktir_mod.
    # make_ktir_mod runs _make_ttir again internally — that's idempotent after
    # one pass, so it's fine.
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete_on_close=False) as f:
        f.write(ttir_text)
        f.flush()
        mod = make_ktir_mod(f.name, grid=grid)
    ktir_text = str(mod)

    return clean_ir(ttir_text), clean_ir(ktir_text)


# ---------------------------------------------------------------------------
# load_driver  (from dump_round_trip.py) — import a kernels/<name>/lower.py
# driver declaring KERNEL / SIGNATURE / CONSTEXPRS / [GRID].
# ---------------------------------------------------------------------------

def load_driver(path: Path) -> tuple[str, dict]:
    """Import an external driver file and return ``(key, entry)``.

    The driver may live anywhere on disk. We add its parent directory to
    ``sys.path`` so its own imports resolve, and walk a few levels up looking
    for a ``kernels/`` or ``pyproject.toml`` marker so project-rooted imports
    (``from kernels.foo.tensor_descriptor import ...``) work too.
    """
    path = path.resolve()
    sys.path.insert(0, str(path.parent))
    cur = path.parent
    for _ in range(6):
        if (cur / "kernels").is_dir() or (cur / "pyproject.toml").is_file():
            sys.path.insert(0, str(cur))
            break
        if cur.parent == cur:
            break
        cur = cur.parent

    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    missing = [n for n in ("KERNEL", "SIGNATURE", "CONSTEXPRS")
               if not hasattr(mod, n)]
    if missing:
        raise SystemExit(
            f"driver {path} is missing required attribute(s): "
            f"{', '.join(missing)}"
        )

    entry = {
        "kernel_fn":  mod.KERNEL,
        "signature":  mod.SIGNATURE,
        "constexprs": mod.CONSTEXPRS,
    }
    if hasattr(mod, "GRID") and mod.GRID is not None:
        entry["grid"] = tuple(mod.GRID)
    return path.stem, entry


def write_variant(dest_root: Path, key: str, ktir: str) -> None:
    """Write ``<dest_root>/<key>/<key>.ktir`` with a round-trip header.

    The header line matches upstream so ``gen_ktir.py`` strips it consistently
    (it drops a leading run of ``//`` comment lines before adding its own
    provenance header)."""
    folder = dest_root / key
    folder.mkdir(parents=True, exist_ok=True)
    header = f"// Round-trip variant: {key}\n\n"
    (folder / f"{key}.ktir").write_text(header + ktir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--driver", type=Path, action="append", default=[], required=True,
        help="driver file declaring KERNEL/SIGNATURE/CONSTEXPRS/[GRID] "
             "(may be passed multiple times)",
    )
    parser.add_argument(
        "--dest", type=Path,
        default=Path(tempfile.mkdtemp(prefix="spyre_rt_")),
        help="destination folder for the round-trip tree (default: a temp dir)",
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    for driver in args.driver:
        key, entry = load_driver(driver)
        print(f"[{key}] compiling ...", flush=True)
        _ttir, ktir = compile_variant(entry)
        write_variant(args.dest, key, ktir)
        print(f"[{key}] wrote {args.dest / key}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
