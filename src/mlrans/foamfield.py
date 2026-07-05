"""Minimal writer for OpenFOAM volScalarField internal values.

Used by the a-posteriori study to inject a corrected eddy-viscosity field
(nut_corr = beta * nut_baseline) into a case. We reuse an existing field file
as a template (keeping its FoamFile header, dimensions and boundaryField) and
only overwrite the internalField list, so the result is a valid OpenFOAM field
in the mesh's native cell order.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def write_scalar_internal(template: Path, out: Path, values: np.ndarray) -> Path:
    """Write `values` as the internalField of a volScalarField.

    Parameters
    ----------
    template : an existing volScalarField file on the same mesh (e.g. the
               baseline `nut`), providing header/dimensions/boundaryField.
    out      : destination path.
    values   : 1-D array in mesh cell order (length = number of cells).
    """
    text = Path(template).read_text()
    v = np.asarray(values, dtype=float).ravel()
    body = "internalField   nonuniform List<scalar> \n{n}\n(\n{vals}\n)\n;".format(
        n=v.size, vals="\n".join(f"{x:.10g}" for x in v))
    # Replace the existing internalField block (uniform or nonuniform) up to
    # its terminating semicolon, without touching boundaryField.
    new_text, n = re.subn(r"internalField.*?;", body, text, count=1, flags=re.S)
    if n != 1:
        raise ValueError(f"Could not locate an internalField block in {template}")
    out = Path(out)
    out.write_text(new_text)
    return out
