"""Steering arms for a SoRH run: what to steer with, how, and through which proxy port.

An arm is one condition: a name, a mode (none | clamp | add | ablate), a dose (clamp threshold or add
scale in the artifact's own units), a proxy port and a steering artifact (`.safetensors` + `.json`
sidecar; `-` = the canonical u). The table format is the one every experiment's `arms.tsv` uses, so one table can drive the LCB arm runner and this eval:

    # name   mode   dose   port   artifact                                             [upstream]
    base     none   -      8100   -
    v3b_k8   add    0.3333 8101   experiments/opt-vector/artifacts/opt_27b_L31_v3b.safetensors
    u_t-20   clamp  -20    8105   -
    v3c_t-3  clamp  -3     8104   experiments/opt-vector/artifacts/opt_27b_L31_v3c_cl.safetensors  http://localhost:8002

The optional sixth column routes an arm to a second vLLM server on a two-GPU box (the launcher's
UPSTREAM is the default). `read_arms_tsv` accepts 4 to 6 columns; Kyle's fleet reader takes 4 or 5, so
leave the column out of a table meant for both.

The analysis needs (vector, mode, dose) per arm to group a dose sweep by vector. With a table those
come from the table (vector = the artifact's file stem); without one they are parsed from the arm
NAME with the sweep convention `<vec>_k<k>` (add, k units along the unit vector), `<vec>_a<scale>`
(add, raw scale) or `<vec>_t<thr>` (clamp), which is what the run of record used
(`parse_cell_name`). Prefer the table: a name cannot say which file it ran.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

MODES = ("none", "clamp", "add", "ablate")
CANONICAL_U = "src/steering_reasoning/canonical/u_L35_steering.safetensors"

# The six arms of the first SoRH run (LCB doses of record), whose bare names carry no dose:
# name -> (vector label, mode, dose). Kept so the tables of record still parse.
ARMS_OF_RECORD = {
    "base": ("base", "none", ""),
    "v3b": ("v3b add", "add", 0.75),
    "v3c": ("v3c add", "add", 0.5),
    "v3b_cl": ("v3b_cl clamp", "clamp", -5),
    "v3c_cl": ("v3c_cl clamp", "clamp", -8),
    "u_cl8": ("u clamp", "clamp", -8),
}
# Sweep-cell names: <vec>_k<k> | <vec>_a<scale> (add) | <vec>_t<thr> (clamp); vec may itself hold '_'.
CELL_RE = re.compile(r"^(?P<vec>.+?)_(?P<kind>[akt])(?P<dose>-?[0-9]+(?:\.[0-9]+)?)$")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    mode: str = "none"
    dose: float | None = None
    port: int | None = None
    artifact: str | None = None      # repo-relative or absolute path; None for mode none
    upstream: str | None = None      # vLLM server URL override for this arm (launcher default otherwise)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"{self.name}: mode must be one of {MODES}, got {self.mode!r}")
        if (self.mode in ("clamp", "add")) != (self.dose is not None):
            raise ValueError(f"{self.name}: {self.mode} needs a dose" if self.dose is None
                             else f"{self.name}: mode {self.mode} takes no dose")
        if self.dose is not None and not math.isfinite(self.dose):
            raise ValueError(f"{self.name}: dose must be finite")
        if self.mode != "none" and not self.artifact:
            raise ValueError(f"{self.name}: mode {self.mode} needs an artifact")
        if self.mode == "none" and self.artifact:
            raise ValueError(f"{self.name}: mode none takes no artifact")

    @property
    def steered(self) -> bool:
        return self.mode != "none"

    @property
    def vector(self) -> str:
        """Group label for tables: 'base' for an unsteered arm, else '<artifact stem> <mode>'."""
        if not self.steered:
            return "base"
        return f"{Path(self.artifact).stem} {self.mode}"

    def dose_label(self):
        return "" if self.dose is None else (int(self.dose) if float(self.dose).is_integer() else self.dose)

    def proxy_args(self, cap: int | None = None, cache_salt: bool = True, root: str | Path | None = None) -> list[str]:
        """Arguments for `python -m steering_reasoning.proxy` (after --port/--upstream)."""
        args: list[str] = []
        if cap is not None:
            args += ["--max-tokens", str(cap)]
        if cache_salt:
            args += ["--cache-salt", self.name]
        if self.mode == "none":
            args.append("--no-steer")
        else:
            art = self.artifact
            if root is not None and not Path(art).is_absolute():
                art = str(Path(root) / art)
            args += ["--mode", self.mode, "--artifact", art]
            if self.mode == "clamp":
                args += ["--threshold", str(self.dose)]
            elif self.mode == "add":
                args += ["--scale", str(self.dose)]
        return args

    def resolved(self, root: str | Path) -> "ArmSpec":
        """The same arm with a relative artifact path made absolute under `root`."""
        if self.artifact and not Path(self.artifact).is_absolute():
            return replace(self, artifact=str(Path(root) / self.artifact))
        return self


def _dose(s: str) -> float | None:
    if s in ("-", "", "none", "None"):
        return None
    return float(s)


def read_arms_tsv(path: str | Path, default_artifact: str = CANONICAL_U) -> list[ArmSpec]:
    """Parse an arms table (whitespace-separated; `#` comments; 4 to 6 columns, see the module doc).
    `-` or a missing artifact means `default_artifact` (the canonical u) for a steered arm."""
    arms, names, ports = [], set(), {}
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) < 4 or len(fields) > 6:
            raise ValueError(f"{path}:{lineno}: expected 'name mode dose port [artifact [upstream]]', got {line!r}")
        name, mode, dose, port, *rest = fields
        if name in names:
            raise ValueError(f"{path}:{lineno}: duplicate arm name {name!r}")
        names.add(name)
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError(f"{path}:{lineno}: bad port {port!r}")
        if int(port) in ports:
            raise ValueError(f"{path}:{lineno}: port {port} already used by arm {ports[int(port)]!r}")
        ports[int(port)] = name
        artifact = rest[0] if rest and rest[0] != "-" else None
        upstream = rest[1] if len(rest) > 1 and rest[1] != "-" else None
        if mode != "none" and artifact is None:
            artifact = default_artifact
        if mode == "none":
            artifact = None
        arms.append(ArmSpec(name=name, mode=mode, dose=_dose(dose), port=int(port), artifact=artifact, upstream=upstream))
    return arms


def parse_arm_spec(spec: str) -> ArmSpec:
    """One `name:port:upstream:mode:artifact:dose` entry (the launcher's ARM_SPECS form; mode none =
    passthrough, artifact and dose empty). The upstream is a URL or a bare port number."""
    parts = spec.split(":")
    if len(parts) < 4:
        raise ValueError(f"expected name:port:upstream:mode[:artifact[:dose]], got {spec!r}")
    # the upstream may itself contain ':' (http://host:port); take name, port from the front and
    # mode, artifact, dose from the back
    name, port = parts[0], parts[1]
    tail = parts[2:]
    # the mode is the FIRST field after the port that is a mode word (a bare port or an http URL never is);
    # what follows is artifact[:dose], and an artifact path may itself contain ':'
    idx = next((i for i, p in enumerate(tail) if p in MODES), None)
    if idx is None:
        raise ValueError(f"no mode in {spec!r}")
    upstream = ":".join(tail[:idx]) or None
    if upstream and upstream.isdigit():
        upstream = f"http://localhost:{upstream}"
    mode = tail[idx]
    rest = tail[idx + 1:]
    if len(rest) >= 2:
        artifact, dose = ":".join(rest[:-1]) or None, _dose(rest[-1])
    elif len(rest) == 1:
        artifact, dose = rest[0] or None, None
    else:
        artifact, dose = None, None
    return ArmSpec(name=name, mode=mode, dose=dose, port=int(port), artifact=artifact, upstream=upstream)


def parse_cell_name(name: str):
    """(vector label, mode, dose) from a name alone. Arms of record first; then the sweep
    convention; else ('?', '?', ''). `u_cl_t-12` -> ('u clamp', 'clamp', -12); `v3b_k5` ->
    ('v3b add', 'add', 5); `myvec_t-3` -> ('myvec clamp', 'clamp', -3); base -> ('base', 'none', '')."""
    if name in ARMS_OF_RECORD:
        return ARMS_OF_RECORD[name]
    m = CELL_RE.match(name)
    if not m:
        return "?", "?", ""
    dose = float(m.group("dose"))
    if dose == int(dose):
        dose = int(dose)
    mode = "add" if m.group("kind") in ("a", "k") else "clamp"
    vec = m.group("vec")
    if vec == "u_cl":
        vec = "u"
    return f"{vec} {mode}", mode, dose


def arm_labels(name: str, arms: dict[str, ArmSpec] | list[ArmSpec] | None = None):
    """(vector, mode, dose) for a table row: from the arms table when the arm is in it, else from the name."""
    if arms:
        table = arms if isinstance(arms, dict) else {a.name: a for a in arms}
        a = table.get(name)
        if a is not None:
            return a.vector, a.mode, a.dose_label()
    return parse_cell_name(name)


def mildness(mode: str, dose) -> float:
    """Smaller = milder. add: |scale| (an addition arm may carry a negative scale, e.g. Kyle's h + s*u with
    s = -4); clamp: minus the threshold (a higher threshold binds less); none/ablate: 0."""
    d = float(dose) if dose not in (None, "") else 0.0
    if mode == "add":
        return abs(d)
    if mode == "clamp":
        return -d
    return 0.0


# The vector groups of the SoRH run of record, in the order its tables list them.
VECTORS_OF_RECORD = ["base", "v3b add", "v3c add", "v3b_cl clamp", "v3c_cl clamp", "u clamp"]
