import yaml

with open("/home/ulgen/Documents/Python_Projects/Contradiction/configurations.yaml", 'r') as stream:
    try:
        configs = (yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

print(configs["main_path"])