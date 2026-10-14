"""Current RQ3 entry: reproduce physical outcomes on all three platforms."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'publication/reproduce_tables.py'), run_name='__main__')
