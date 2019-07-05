import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from asv_to_pandas import create_benchmark_dataframe

from benchmarks.config import N_SAMPLES


df = create_benchmark_dataframe(group_by="class")
## plot the distributions of the benchmarks
all_benchmarks.hist()
plt.show()
all_benchmarks = {}
for name, res in df.items():
    estimator, bench_type = re.findall(r'(\w+)(SingleFit|MultipleFit)Bench', s)[0]
    n_samples = N_SAMPLES[estimator]
    all_benchmarks[(estimator, bench_type, n_samples)] = res
all_benchmarks = pd.concat(all_benchmarks, names=('estimator', 'parallel_level', 'effective_n_samples'))
import numpy as np
plot_df = all_benchmarks.xs('2', level='n_jobs').unstack('backend')
f, ax = plt.subplots()
plot_df.index.get_level_values('effective_n_samples')
x = plot_df['threading']
y = plot_df['loky']
shapes = np.log10(plot_df.index.get_level_values('effective_n_samples'))
ax.scatter(x, y, s=10*shapes, marker='o')
ax.set_xlabel('running time (threading backend)')
ax.set_ylabel('running time (multiprocessing backend)')
