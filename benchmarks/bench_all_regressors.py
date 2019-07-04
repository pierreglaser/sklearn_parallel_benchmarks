#!/usr/bin/env python
# -*- coding: utf-8 -*-
# programatically create benchmark classes for all classifiers.
#
# Author: Pierre Glaser
from .base import make_all_bench_classes


globals().update(make_all_bench_classes(type_filter='regressor'))
