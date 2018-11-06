from collections import defaultdict
import os
import socket

import numpy as np
import pandas as pd

from asv.benchmarks import Benchmarks
from asv.config import Config
from asv.commands.publish import Publish

HOME = os.environ.get('HOME')
hostname = socket.gethostname()


def _remove_quotes(params):
    # params is a list of lists, each list contains the values of one parameter
    unquoted_params_values = []
    for param_values in params:
        unquoted_param_values = []
        for param_value in param_values:
            unquoted_param_values.append(param_value.replace("'", ""))
        unquoted_params_values.append(unquoted_param_values)
    return unquoted_params_values


def create_benchmark_dataframe(group_by='name'):
    repo_dirname = os.path.dirname(__file__)
    config_path = os.path.join(repo_dirname, 'asv.conf.json')
    config = Config.load(config_path)

    benchmarks = Benchmarks.load(config)

    results = defaultdict(dict)
    metadata_levels = ['type', 'name', 'class', 'file', 'version',
                       'commit_hash', 'date']

    if isinstance(group_by, str):
        group_by = [group_by]
    levels_to_group_by = group_by

    levels_to_concat_on = [l for l in metadata_levels if l not in
                           levels_to_group_by]

    for single_env_result in Publish.iter_results(config, benchmarks):
        benchmark_metadata = {
            'version': single_env_result._params['python'],
            'commit_hash': single_env_result._commit_hash,
            'date': single_env_result._date}

        for b_name, params in single_env_result._benchmark_params.items():
            unquoted_params = _remove_quotes(params)
            filename, classname, benchname = b_name.split('.')

            _benchmark = benchmarks[b_name]
            b_type, param_names = _benchmark['type'], _benchmark['param_names']

            benchmark_metadata.update({
                'type': b_type,
                'file': filename,
                'class': classname,
                'name': benchname})

            values_to_group_by = tuple([benchmark_metadata[key] for key in
                                        levels_to_group_by])
            values_to_concat_on = tuple([benchmark_metadata[key] for key in
                                        levels_to_concat_on])

            # this is dangerous because we there is no reason the results
            # order follow the carthesian product of the parameter space,
            # however empirically it seems to be the case
            mi = pd.MultiIndex.from_product(unquoted_params, names=param_names)
            _results = pd.Series(single_env_result._results[b_name], index=mi)

            results[values_to_group_by][values_to_concat_on] = _results

    clean_result = {}
    for k, v in results.items():
        if len(k) == 1:
            # if key if a list of length one, convert it to a string by taking
            # its only element
            clean_result[k[0]] = pd.concat(v, names=levels_to_concat_on)
        elif len(k) == 0:
            # if key is of length 0, there is only one element, so, return the
            # underlying dict
            clean_result = pd.concat(v, names=levels_to_concat_on)
        else:
            clean_result[k] = pd.concat(v, names=levels_to_concat_on)

    return clean_result


if __name__ == "__main__":
    all_bench = create_benchmark_dataframe(group_by='class')
