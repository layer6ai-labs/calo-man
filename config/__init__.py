import ast

from .get_config import get_configs, get_epl_configs, load_configs, load_epl_configs, load_config, get_single_config, get_dimension_config, load_configs_from_run_dir, load_config_from_run_dir


def parse_config_arg(key_value):
    assert "=" in key_value, "Must specify config items with format `key=value`"

    k, v = key_value.split("=", maxsplit=1)

    assert k, "Config item can't have empty key"
    assert v, "Config item can't have empty value"

    try:
        v = ast.literal_eval(v)
    except ValueError:
        v = str(v)

    return k, v
