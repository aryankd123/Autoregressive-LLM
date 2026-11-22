import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                    # tests import each other's fixtures
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # repo root -> import gpt_from_scratch
