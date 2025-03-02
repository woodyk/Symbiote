#!/usr/bin/env python3
#
# term-plotlib.py

import termplotlib as tpl
import numpy as np

rng = np.random.default_rng(123)
sample = rng.standard_normal(size=1000)
counts, bin_edges = np.histogram(sample)

fig = tpl.figure()
fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
fig.show()
