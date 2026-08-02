# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'SHHA':
        from crowd_datasets.SHHA.loading_data import loading_data
        return loading_data
    if args.dataset_file == 'VisDrone':
        from crowd_datasets.VisDrone.loading_data import loading_data
        return lambda data_root: loading_data(data_root, args)

    return None
