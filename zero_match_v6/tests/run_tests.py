# -*- coding: utf-8 -*-
"""Запуск всех тестов без pytest: python tests/run_tests.py"""
import glob
import importlib.util
import os
import sys
import traceback

failed = total = 0
for path in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "test_*.py"))):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, fn in list(vars(mod).items()):
        if name.startswith("test_") and callable(fn):
            total += 1
            try:
                fn()
                print(f"PASS {mod.__name__}::{name}")
            except Exception:
                failed += 1
                print(f"FAIL {mod.__name__}::{name}")
                traceback.print_exc()
print(f"\n{total - failed}/{total} passed")
sys.exit(1 if failed else 0)
