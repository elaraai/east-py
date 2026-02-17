#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Setuptools build script with optional Cython acceleration.

Discovers all .pyx files under east/ and compiles them to C extensions.
If compilation fails (e.g. no C compiler), falls back to pure Python
with no error — the import shims in each .py module handle the fallback.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """build_ext that treats compilation failure as non-fatal.

    If the C compiler is unavailable or compilation fails for any reason,
    the package installs as pure Python. The Cython import shims (e.g.
    ``with contextlib.suppress(ImportError): from ._foo_cy import ...``)
    will simply not find the .so files and use the Python fallback.
    """

    def run(self):
        try:
            super().run()
        except Exception as exc:
            print(f"Warning: Cython extension compilation failed ({exc}), using pure Python")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:
            print(f"Warning: Failed to compile {ext.name} ({exc}), skipping")


def get_ext_modules():
    """Discover .pyx files and cythonize them."""
    try:
        from Cython.Build import cythonize
    except ImportError:
        print("Warning: Cython not available, skipping extension compilation")
        return []

    east_dir = Path(__file__).parent / "east"
    pyx_files = sorted(east_dir.rglob("*.pyx"))

    if not pyx_files:
        return []

    package_root = Path(__file__).parent
    extensions = []
    for pyx_path in pyx_files:
        rel_path = pyx_path.relative_to(package_root)
        module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
        extensions.append(Extension(module_name, [str(rel_path)]))

    exts = cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
        },
    )
    # cythonize() can produce absolute paths for generated .c files,
    # which breaks wheel builds from sdists. Relativize them.
    for ext in exts:
        ext.sources = [os.path.relpath(s, package_root) for s in ext.sources]
    return exts


setup(
    ext_modules=get_ext_modules(),
    cmdclass={"build_ext": OptionalBuildExt},
)
