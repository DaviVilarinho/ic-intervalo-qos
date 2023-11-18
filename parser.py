import argparse

from sparsing import NaiveSparseStrategy

def parse_strategy_and_sparsing_factor():
    parser = argparse.ArgumentParser(description='strategy and sparsing-factor parser')

    parser.add_argument('--strategy', required=True, help='Specify the strategy: Naive, ')

    parser.add_argument('--sparsing-factor', required=True, type=float, help='Specify the sparsing factor (for each SF records pick one...)')

    args = parser.parse_args()

    return NaiveSparseStrategy() if args.strategy == 'Naive' else None, args.sparsing_factor

