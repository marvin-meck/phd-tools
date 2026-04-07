"""phdtools.__init__.py

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

import configparser
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default paths
DEFAULT_DATA_DIR = PROJECT_ROOT / "phd-data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "phd-results"
DEFAULT_TMP_DIR = PROJECT_ROOT / "tmp"


# Load config
def load_config():
    config = configparser.ConfigParser()

    config_paths = [
        os.environ.get("PHDTOOLS_CONFIG"),  # explicit override
        PROJECT_ROOT / "config.ini",
        Path.home() / ".config" / "phdtools" / "config.ini",
    ]

    for path in config_paths:
        if path and Path(path).exists():
            config.read(path)
            break

    return config


config = load_config()


# Set paths
def resolve_path(env_var, section, key, default):
    # 1. environment variable
    if env_var in os.environ:
        return Path(os.environ[env_var]).expanduser().resolve()

    # 2. config file
    if config.has_option(section, key):
        return Path(config.get(section, key)).expanduser().resolve()

    # 3. fallback
    return default


DATA_DIR = resolve_path("PHDTOOLS_DATA_DIR", "paths", "data_dir", DEFAULT_DATA_DIR)
RESULTS_DIR = resolve_path(
    "PHDTOOLS_RESULTS_DIR", "paths", "results_dir", DEFAULT_RESULTS_DIR
)
TMP_DIR = resolve_path("PHDTOOLS_TMP_DIR", "paths", "tmp_dir", DEFAULT_TMP_DIR)

# Ensure directories exist
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
