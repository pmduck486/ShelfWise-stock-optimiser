"""Benchmark one full-size origin: python -m examples.run_full."""

from .run_small import main


if __name__ == '__main__':
    main(default_profile='full')
