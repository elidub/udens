def print_dict(d, indent=0):
    instance = type(d)
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, instance):
            print_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))