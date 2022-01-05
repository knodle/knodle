"""Define argument parser class."""
import argparse
from pathlib import Path


class ArgParser(object):
    """Argument parser for label.py"""
    def __init__(self):
        """Initialize argument parser."""
        parser = argparse.ArgumentParser()

        # # Input report parameters.
        # parser.add_argument('--reports_path',
        #                     required=True,
        #                     help='Path to file with radiology reports.')

        # Output parameters.
        # parser.add_argument('--output_path',
        #                     default='labeled_reports.csv',
        #                     help='Output path to write labels to.')

        # Misc.
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            help='Print progress to stdout.')

        self.parser = parser

    def parse_args(self):
        """Parse and validate the supplied arguments."""
        args = self.parser.parse_args()

        #args.reports_path = Path(args.reports_path)
        #args.output_path = Path(args.output_path)

        return args
