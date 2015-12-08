#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from test_utils import TestUtils


def run_test():
    test_classes = [TestUtils]

    loader = unittest.TestLoader()
    suites_list = []

    for test_class in test_classes:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    suites = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    runner.run(suites)

if __name__ == '__main__':
    run_test()
