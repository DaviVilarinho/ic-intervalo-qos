import argparse
from replication import IS_LOCAL

from sparsing import NaiveSparseStrategy

def parse_strategy_and_sparsing_factor():
    parser = argparse.ArgumentParser(description='strategy and sparsing-factor parser')

    parser.add_argument('--strategy', required=False, default='Naive', type=str, help='Specify the strategy: Naive, ')

    parser.add_argument('--sparsing-factor', required=False, default=1, type=int, help='Specify the sparsing factor (for each SF records pick one...)')

    parser.add_argument('--destination', required=False, default='/tmp/exp' if IS_LOCAL else None, help='Specify the sparsing factor (for each SF records pick one...)')

    args = parser.parse_args()

    return NaiveSparseStrategy() if args.strategy == 'Naive' else None, \
        args.sparsing_factor, \
        args.destination


