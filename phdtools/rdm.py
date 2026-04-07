"""phdtools.rdm.py

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

import csv
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
import os
from pathlib import Path
import warnings
from functools import wraps

from sqids import Sqids

from phdtools import RESULTS_DIR

MAX_ID = 1000
AUTHOR = "Marvin Meck"
ORCID = "https://orcid.org/0000-0003-2930-8220"


INDEX = RESULTS_DIR / "index.csv"


class DataType(IntEnum):
    SUPPORT = 0
    FIGURE = 1
    TABLE = 2


class Chapter(IntEnum):
    UNUSED = 0
    INTRODUCTION = 1
    LITERATURE_REVIEW = 2
    METHODS = 3
    RESULTS = 4
    DISCUSSION = 5
    CONCLUSION = 6
    APPENDIX = 7


sqids_encoder = Sqids()


@dataclass(frozen=True)
class DataID:
    type: DataType
    chapter: Chapter
    counter: int

    def __post_init__(self):
        if self.counter <= 0:
            raise ValueError(f"Counter must be greater than zero.")

    def to_tuple(self) -> tuple[int, int, int]:
        return (int(self.type), int(self.chapter), self.counter)

    def to_sqid(self) -> str:
        return sqids_encoder.encode(self.to_tuple())

    @classmethod
    def from_sqid(cls, sqid: str) -> "DataID":
        parts = sqids_encoder.decode(sqid)
        if len(parts) != 3:
            raise ValueError(f"Invalid Sqid: {sqid}")
        return cls(DataType(parts[0]), Chapter(parts[1]), parts[2])

    def get_path(self, base_dir: Path = RESULTS_DIR, fail_exists=True) -> Path:
        path = base_dir / self.to_sqid()
        if path.exists() and fail_exists:
            raise FileExistsError(f"Results path already exists: {path}")
        return path

    def ensure_unique_path(self, base_dir: Path = RESULTS_DIR) -> Path:
        """Like get_path(), but raises if path exists."""
        return self.get_path(base_dir, fail_exists=True)

    def __str__(self):
        return f"{self.type.name.lower()}-ch{self.chapter.value}-{self.counter:02d}"


def request_free_id(
    data_type: DataType,
    chapter: Chapter,
    base_dir: Path = RESULTS_DIR,
    start_counter: int = 1,
    max_attempts: int = MAX_ID,
) -> DataID:
    """
    Finds the next available DataID (i.e. no existing results folder) for the given type and chapter.
    Returns None if no free ID is found within the attempt limit.
    """
    for counter in range(start_counter, start_counter + max_attempts):
        candidate = DataID(data_type, chapter, counter)
        path = candidate.get_path(base_dir, fail_exists=False)
        if not path.exists():
            return candidate
    raise ValueError("No available IDs found. Too many results? Try setting MAX_ID.")
    # return None  # too many attempts; all used


def update_index(data_id, doc="", index=INDEX, _tmp="_tmp.csv"):
    if not os.path.exists(index):
        with open(index, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "type", "chapter", "counter", "description"])

    new = [data_id.to_sqid(), *data_id.to_tuple(), doc]
    with open(index, newline="") as infile:
        reader = csv.reader(infile)
        with open(_tmp, "w", newline="") as outfile:
            writer = csv.writer(outfile)
            for row in reader:
                if row[0] != data_id.to_sqid():
                    writer.writerow(row)
            writer.writerow(new)
        os.remove(index)
        os.rename(_tmp, index)


def auto_create_path(func):
    @wraps(func)
    def wrapper(data_id, doc, *args, overwrite=False, **kwargs):

        path = data_id.get_path(fail_exists=False)

        if path.exists() and not overwrite:
            warnings.warn(
                f"Results path '{path}' already exists. The data_id has been used before. "
                f"This may be because the results already exist. Skipping!"
            )
            return None

        # register data_id in the index
        update_index(data_id=data_id, doc=doc)

        # Create or re-create the path if overwrite=True
        path.mkdir(parents=True, exist_ok=True)

        return func(path, *args, **kwargs)

    return wrapper


def write_metadata(ostream, description=None, comment="#"):
    ostream.write(f"{comment} METADATA\n")

    ostream.write(f"{comment} Author: {AUTHOR}\n")
    ostream.write(f"{comment} ORCiD: {ORCID}\n")
    ostream.write(f"{comment} Date: {datetime.today()}\n")

    if not description is None:
        ostream.write(f"{comment} Description: ")
        for num, line in enumerate(description.splitlines()):
            if num > 0:
                ostream.write(f"{comment}   {line}\n")
            else:
                ostream.write(f"{line}\n")

    ostream.write(f"{comment} END\n")
