import yaml
import os

def get_config():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'config.yaml')
    
    with open(config_path, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
    
    return config_data

config = get_config()

def get_this_config(this):
    if this not in config:
        raise ValueError("Key {} not found in config".format(this))
    return config[this]

def get_mount_config():
    return config['mount']

def get_dataloader_config():
    return config['dataloader']

def deparse_args(config, what):
    if not isinstance(what, str):
        raise TypeError("Expected string, got {}".format(type(what)))
    if what not in config:
        raise ValueError("Key {} not found in config".format(what))
    args = config[what]
    if not isinstance(args, dict):
        raise TypeError("Expected dict, got {}".format(type(args)))
    
    try:
        arg_str = ""
        for key, value in args.items():
            if not isinstance(key, str):
                raise TypeError("Expected string, got {}".format(type(key)))
            if isinstance(value, bool):
                arg_str += " --{}".format(key)
            elif isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
                arg_str += " --{} {}".format(key, value)
            else:
                raise TypeError("Expected string, int, float or bool, got {}".format(type(value)))
    except Exception as e:
        raise yaml.YAMLError("Error while parsing args: {}".format(e))
    
    return arg_str
    
