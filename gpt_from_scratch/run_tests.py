"""Run every test without needing pytest:

    python -m gpt_from_scratch.run_tests            # all parts
    python -m gpt_from_scratch.run_tests 3 4        # only test_part3_* and test_part4_* (attention)

(pytest works too:  python -m pytest gpt_from_scratch/tests -q)
"""

import importlib
import inspect
import os
import sys
import time
import traceback

TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    sys.path.insert(0, TESTS)
    sys.path.insert(0, os.path.dirname(os.path.dirname(TESTS)))
    files = sorted(f for f in os.listdir(TESTS) if f.startswith("test_") and f.endswith(".py"))
    if argv:
        files = [f for f in files if any(f.startswith(f"test_part{p}") for p in argv)]
    passed = failed = 0
    for f in files:
        mod = importlib.import_module(f[:-3])
        print(f"\n{f}")
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__:
                continue
            t = time.time()
            try:
                fn()
                passed += 1
                print(f"  PASS  {name}  ({time.time() - t:.2f}s)")
            except Exception:
                failed += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
