"""scripts/codata.py

Copyright 2025 Marvin Meck

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import re

import requests

URL = r"https://physics.nist.gov/cuu/Constants/Table/allascii.txt"

WANTED_EXACT = {
    "boltzmann constant",
    "avogadro constant",
    "elementary charge",
}

# Lines to skip explicitly (headers, titles, sources, urls, rules, etc.)
SKIP = re.compile(
    r"""
    ^\s*$                               |  # empty
    Fundamental\ Physical\ Constants    |  # title
    ^\s*\d{4}\s+CODATA\b                |  # e.g., '2022 CODATA adjustment'
    ^\s*From:                           |  # source line
    ^\s*Quantity\b                      |  # column header
    ^\s*[-=]{5,}\s*$                       # horizontal rule
    """,
    re.X,
)

ELLIPSIS = re.compile(r"(?:\.{3}|…)")
WEIRD_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")

# Column separator: 3+ spaces or tabs (safer than 2)
SEP = r"[ \t]{3,}"

# Numbers: digits/dots with optional internal spaces and optional exponent (space before 'e' allowed)
NUM = r"[+\-]?[0-9.](?:[0-9. ]*[0-9.])?(?:\s*[eE][+\-]?\d+)?"

# Strict data row: quantity, value, uncertainty, optional unit
ROW = re.compile(
    rf"""^
    (?P<quantity>\S.+?) {SEP}                     # quantity
    (?P<value>{NUM}) {SEP}                        # value
    (?P<uncertainty>(?:{NUM}|\(exact\)))          # uncertainty (number or "(exact)")
    (?: {SEP} (?P<unit>\S.*?))?                   # optional unit
    \s*$
    """,
    re.X,
)


def _to_float(num_str: str) -> float:
    s = re.sub(r"\s+", "", num_str).replace("−", "-")
    return float(s)


def _to_float_or_exact(s: str):
    s = s.strip()
    if s.lower() == "(exact)":
        return None
    return _to_float(s)


def _norm_name(s: str) -> str:
    # collapse whitespace, strip, lowercase
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_codata_ascii(text: str):
    out = {}
    for raw in text.splitlines():
        line = WEIRD_SPACES.sub(" ", raw).rstrip()
        if SKIP.search(line):
            continue

        m = ROW.match(line)
        if not m:
            continue

        # Skip truncated/derived rows that show ellipses
        if ELLIPSIS.search(m.group("value")) or ELLIPSIS.search(m.group("uncertainty")):
            continue

        q_raw = m.group("quantity").strip()
        q_norm = _norm_name(q_raw)

        # Only accept exact target names
        if q_norm not in WANTED_EXACT:
            continue

        val = _to_float(m.group("value"))
        unc = _to_float_or_exact(m.group("uncertainty"))
        unit = (m.group("unit") or "").strip()

        out[q_raw] = {"value": val, "uncertainty": unc, "unit": unit, "raw_name": q_raw}
    return out


def main(project_root, url=URL, fname=None):

    if (url is None) and (fname is None):
        raise ValueError("Either url or fname must be set!")

    if not url is None:
        r = requests.get(url)

        if r.ok:
            constants = parse_codata_ascii(r.text)
            with open(project_root / "phdtools" / "data" / "constants.py", "w+") as f:
                # with StringIO() as f:
                f.write('"""Generated source file."""\n')
                f.write(
                    f"AVOGADRO_CONST_SI = {constants["Avogadro constant"]["value"]} # {constants["Avogadro constant"]["unit"]}\n"
                )
                f.write(
                    f"BOLTZMANN_CONST_SI = {constants["Boltzmann constant"]["value"]} # {constants["Boltzmann constant"]["unit"]}\n"
                )
                f.write(
                    f"ELEMENTARY_CHARGE_SI = {constants["elementary charge"]["value"]} # {constants["elementary charge"]["unit"]}\n"
                )
                f.write("FARADAY_CONST_SI = ELEMENTARY_CHARGE_SI * AVOGADRO_CONST_SI\n")
                f.write("GAS_CONST_SI = BOLTZMANN_CONST_SI * AVOGADRO_CONST_SI\n")
                # print(f.getvalue())
    elif not fname is None:
        # TODO: local file option
        raise NotImplementedError(
            "Creating the source file from a local copy of the text file is not yet implemented."
        )
    else:
        raise ValueError("Can't be reached")


if __name__ == "__main__":
    main()
