"""Define argument parser class."""
import argparse
from pathlib import Path


class ArgParser(object):
    """Argument parser for label.py"""
    def __init__(self):
        """Initialize argument parser."""
        parser = argparse.ArgumentParser()

        # Misc.
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            help='Print progress to stdout.')

        self.parser = parser

    def parse_args(self):
        """Parse and validate the supplied arguments."""
        args = self.parser.parse_args()

        return args
