#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple


Data = namedtuple('Data', ['X', 'Y', 'y', ])

Dataset = namedtuple('Dataset', ['unlabel', 'train', 'test', ])
