#!/usr/bin/env bash
set -euo pipefail
python main.py --model densenet --mode all "$@"
