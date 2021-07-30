# ReverbDataset

AgentOS wrapper around [DeepMind's Reverb](https://github.com/deepmind/reverb)
data storage and transport system.

## Installation

Requires Python 3.6 to 3.8.

* Create a virtualenv, e.g. `virtualenv -p python3.8 env`
* Activate your virtualenv: `source env/bin/activate`
* Clone the latest [AgentOS](https://github.com/agentos-project/agentos) master
* `pip install -e [path/to/agentos/clone/]`
* `pip install -r requirements.txt`
* Format code: `python scripts/format_code.py`
* Lint code: `python scripts/lint_code.py`
