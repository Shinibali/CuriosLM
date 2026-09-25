"""
nanoGPT's "Poor Man's Configurator" -- exec'd, not imported, so it overrides the caller's
globals() directly rather than requiring a config. prefix on every variable:

    >>> exec(open('configurator.py').read())

Positional args are config files, exec'd in order; --key=value args override single keys:
$ python train.py config/override_file.py --batch_size=32
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val) # bool, number, etc; falls back to the raw string
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key])
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
