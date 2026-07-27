#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate ``kernels/<name>/<variant>.ktir`` from each kernel's lowering driver.

This is the single source of truth for the committed KTIR: the ``.ktir`` files
are *generated* from the Triton kernel source, not hand-written. For each kernel
directory that contains a ``lower.py`` driver, this script lowers every variant
declared in that driver to ``<variant>.ktir`` in the same directory.

The output filename mirrors the **source kernel variant**: lowering
``tensor_descriptor.py`` writes ``tensor_descriptor.ktir``; a future
``spyre_aware.py`` would write ``spyre_aware.ktir``. So a kernel with multiple
Spyre implementations gets one ``.ktir`` per implementation, side by side.

Under the hood it drives the vendored round-trip lowering at
``scripts/_spyre/round_trip.py`` (a trimmed copy of the Spyre fork's
``dump_round_trip.py --driver`` path).

A driver (``kernels/<name>/lower.py``) declares a ``VARIANTS`` dict mapping each
variant name (= source module name = output ``.ktir`` stem) to the lowering
inputs; see ``kernels/rms_norm/lower.py`` for the canonical example::

    VARIANTS = {
        "tensor_descriptor": {
            "KERNEL":     <@triton.jit function>,
            "SIGNATURE":  {arg: triton-type, ...},
            "CONSTEXPRS": {arg: value, ...},
            "GRID":       [..],     # optional
        },
        # "spyre_aware": {...},
    }

Usage::

This needs the **spyre-enabled** Triton build (stock PyPI Triton has no spyre
backend). Rather than install it persistently, layer it in for the single
generation run with ``uv run --with``, which uses a separate ephemeral env and
leaves the project ``.venv`` (stock PyPI Triton) untouched::

    # Regenerate every variant of every kernel with a lower.py driver:
    # (Proposal 2: PR #19 fork carries the tl.spyre_tensor_layout builtin.)
    GIT_PAT=$(gh auth token) uv run \
        --with "triton @ git+https://github.com/fabianlim/triton@<sha>" \
        python scripts/gen_ktir.py

    # ... one kernel / one variant / CI drift guard:
    ... python scripts/gen_ktir.py rms_norm
    ... python scripts/gen_ktir.py rms_norm:tensor_descriptor
    ... python scripts/gen_ktir.py --check

"""

import argparse
import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = ROOT / "kernels"

# The vendored round-trip lowering module, invoked as `-m scripts._spyre.round_trip`.
_ROUND_TRIP_MODULE = "scripts._spyre.round_trip"


def _require_spyre_backend() -> None:
    """Fail early with an actionable message if the spyre Triton backend
    isn't importable (i.e. run in the base venv without the --with layer)."""
    try:
        import triton.backends.spyre.compiler  # noqa: F401,PLC0415
    except ImportError as exc:
        raise SystemExit(
            "KTIR generation needs the spyre-enabled Triton build, which is "
            "not present in this environment.\n"
            f"(import error: {exc})"
        )


def _variant_names(kernel_name: str) -> list[str]:
    """Return the variant names declared in ``kernels/<name>/lower.py``.

    Parsed statically with ``ast`` (no import) so this works in the base-tier
    venv too — the driver imports triton, which we don't want to require just
    to enumerate variants. Reads the keys of a module-level ``VARIANTS`` dict.
    """
    driver = KERNELS_DIR / kernel_name / "lower.py"
    if not driver.is_file():
        raise SystemExit(f"no driver: {driver}")

    tree = ast.parse(driver.read_text(), filename=str(driver))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "VARIANTS"
                   for t in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            raise SystemExit(f"{driver}: VARIANTS must be a dict literal")
        names = []
        for key in node.value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise SystemExit(
                    f"{driver}: VARIANTS keys must be string literals"
                )
            names.append(key.value)
        if not names:
            raise SystemExit(f"{driver}: VARIANTS is empty")
        return names

    raise SystemExit(
        f"{driver}: no module-level VARIANTS dict found. See "
        "kernels/rms_norm/lower.py for the expected driver shape."
    )


def _provenance_header(kernel_name: str, variant: str) -> str:
    """Header prepended to every generated <variant>.ktir."""
    return (
        f"// Generated from kernels/{kernel_name}/{variant}.py by "
        f"scripts/gen_ktir.py\n"
        f"// Source: kernels/{kernel_name}/lower.py -> VARIANTS['{variant}'].\n"
        f"// DO NOT EDIT BY HAND — regenerate with:\n"
        f"//     .venv/bin/python scripts/gen_ktir.py {kernel_name}:{variant}\n\n"
    )


def discover_kernels() -> list[str]:
    """Kernel names (directory names) that ship a ``lower.py`` driver."""
    return sorted(
        p.parent.name
        for p in KERNELS_DIR.glob("*/lower.py")
    )


# A throwaway driver that re-exports one VARIANTS entry as the flat
# KERNEL/SIGNATURE/CONSTEXPRS/GRID globals round_trip.py's --driver path wants.
# Written into the kernel dir so `from kernels.<name>.lower import VARIANTS`
# resolves the same way the driver's own imports do.
_REEXPORT_TEMPLATE = '''\
# SPDX-License-Identifier: Apache-2.0
# Auto-generated by scripts/gen_ktir.py — do not commit.
from kernels.{kernel}.lower import VARIANTS

_v = VARIANTS[{variant!r}]
KERNEL = _v["KERNEL"]
SIGNATURE = _v["SIGNATURE"]
CONSTEXPRS = _v["CONSTEXPRS"]
GRID = _v.get("GRID")
'''


def compile_variant(kernel_name: str, variant: str) -> str:
    """Lower one ``VARIANTS[variant]`` entry to KTIR; return the file text
    (provenance header + cleaned KTIR body)."""
    kernel_dir = KERNELS_DIR / kernel_name

    env = dict(os.environ)
    # Both nvidia and spyre drivers report active; pin the runtime driver
    # so Triton's auto-selection doesn't raise. (See docs/spyre-triton-build.md.)
    env.setdefault("TRITON_DEFAULT_BACKEND", "nvidia")

    # Re-export driver lives in the kernel dir; its stem is the round_trip key
    # and the produced subfolder/file name.
    drv_stem = f"_genktir_{variant}"
    drv_path = kernel_dir / f"{drv_stem}.py"
    drv_path.write_text(
        _REEXPORT_TEMPLATE.format(kernel=kernel_name, variant=variant)
    )

    try:
        with tempfile.TemporaryDirectory(prefix="gen_ktir_") as tmp:
            # Same interpreter we're running under (the .venv python with the
            # spyre Triton). cwd=ROOT so `-m scripts._spyre.round_trip` and the
            # driver's project-rooted imports resolve.
            proc = subprocess.run(
                [
                    sys.executable, "-m", _ROUND_TRIP_MODULE,
                    "--driver", str(drv_path),
                    "--dest", tmp,
                ],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                sys.stderr.write(proc.stdout)
                sys.stderr.write(proc.stderr)
                raise SystemExit(
                    f"[{kernel_name}:{variant}] round-trip lowering failed "
                    f"(exit {proc.returncode})"
                )

            # round_trip writes <dest>/<driver-stem>/<driver-stem>.ktir.
            produced = Path(tmp) / drv_stem / f"{drv_stem}.ktir"
            if not produced.is_file():
                sys.stderr.write(proc.stdout)
                raise SystemExit(
                    f"[{kernel_name}:{variant}] expected output not found: "
                    f"{produced}"
                )
            raw = produced.read_text()
    finally:
        drv_path.unlink(missing_ok=True)
        # round_trip imports the driver, leaving a .pyc behind.
        pycache = kernel_dir / "__pycache__"
        for pyc in pycache.glob(f"{drv_stem}.*.pyc"):
            pyc.unlink(missing_ok=True)

    # Strip round_trip's own "// Round-trip variant: ..." header lines (and the
    # blank line after them); keep the MLIR body, then add our provenance.
    body = _strip_leading_comment_block(raw)
    return _provenance_header(kernel_name, variant) + body


def _strip_leading_comment_block(text: str) -> str:
    """Drop a leading run of ``//`` comment lines and following blanks."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("//"):
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "".join(lines[i:])


def _resolve_targets(args_kernels: list[str]) -> list[tuple[str, str]]:
    """Expand CLI args into a list of (kernel, variant) pairs.

    Each arg is either ``<kernel>`` (all its variants) or
    ``<kernel>:<variant>`` (one). No args → every variant of every kernel.
    """
    targets: list[tuple[str, str]] = []
    if not args_kernels:
        for kernel in discover_kernels():
            for variant in _variant_names(kernel):
                targets.append((kernel, variant))
        return targets

    for arg in args_kernels:
        if ":" in arg:
            kernel, variant = arg.split(":", 1)
            available = _variant_names(kernel)
            if variant not in available:
                raise SystemExit(
                    f"{kernel}: no variant '{variant}' "
                    f"(have: {', '.join(available)})"
                )
            targets.append((kernel, variant))
        else:
            for variant in _variant_names(arg):
                targets.append((arg, variant))
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "kernels", nargs="*",
        help="kernel name(s) or <kernel>:<variant> to regenerate "
             "(default: every variant of every kernel with a lower.py)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="don't write; exit non-zero if any committed <variant>.ktir is "
             "stale relative to freshly generated output",
    )
    args = parser.parse_args()

    targets = _resolve_targets(args.kernels)
    if not targets:
        print("no kernels with a lower.py driver found", file=sys.stderr)
        return 1

    _require_spyre_backend()

    stale: list[str] = []
    for kernel, variant in targets:
        generated = compile_variant(kernel, variant)
        out_path = KERNELS_DIR / kernel / f"{variant}.ktir"
        label = f"{kernel}:{variant}"

        if args.check:
            current = out_path.read_text() if out_path.is_file() else None
            if current != generated:
                stale.append(label)
                print(f"[{label}] STALE — {variant}.ktir differs from source")
            else:
                print(f"[{label}] ok")
        else:
            out_path.write_text(generated)
            print(f"[{label}] wrote {out_path.relative_to(ROOT)}")

    if args.check and stale:
        print(
            f"\n{len(stale)} stale: {', '.join(stale)}\n"
            f"Regenerate with: .venv/bin/python scripts/gen_ktir.py "
            f"{' '.join(stale)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
