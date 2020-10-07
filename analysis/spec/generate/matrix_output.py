#!/usr/bin/env python
import argparse
import sys
import csv
import json
import subprocess

def map_2d_dict(struct, command):
    assert len(struct.keys()) == 2, 'Supports only 2-d map'
    rowname, colname = struct.keys()
    rowvals, colvals = struct.values()
    return table_command(rowname, rowvals, colname, colvals, command)


def table_command(rowname, rowvals, colname, colvals, command):
    cells = [[f'{rowname}\{colname}'] + list(map(str, colvals))]
    for rowv in rowvals:
        row = [f'{rowv}']
        for colv in colvals:
            commandc = f'echo {rowv} {colv} | ' + command
            print(commandc, file=sys.stderr)
            run = subprocess.run(commandc, shell=True, check=True, capture_output=True)
            row += [run.stdout.decode().strip()]
        cells += [row]
    return cells


class Table:
    """ Stolen from https://github.com/lzakharov/csv2md/blob/master/csv2md/table.py """
    def __init__(self, cells):
        self.cells = cells
        self.widths = list(map(max, zip(*[list(map(len, row)) for row in cells])))

    def markdown(self, center_aligned_columns=None, right_aligned_columns=None):
        def format_row(row):
            return '| ' + ' | '.join(row) + ' |'

        rows = [format_row([cell.ljust(width) for cell, width in zip(row, self.widths)]) for row in self.cells]
        separators = ['-' * width for width in self.widths]

        if right_aligned_columns is not None:
            for column in right_aligned_columns:
                separators[column] = ('-' * (self.widths[column] - 1)) + ':'
        if center_aligned_columns is not None:
            for column in center_aligned_columns:
                separators[column] = ':' + ('-' * (self.widths[column] - 2)) + ':'

        rows.insert(1, format_row(separators))

        return '\n'.join(rows)

    @staticmethod
    def parse_csv(file, delimiter=',', quotechar='"'):
        return Table(list(csv.reader(file, delimiter=delimiter, quotechar=quotechar)))


def main():
    parser = argparse.ArgumentParser(description='Parse CSV files into Markdown tables.')
    parser.add_argument('command', metavar='COMMAND'
                        , help='Command to run')
    parser.add_argument('-c', '--center-aligned-columns', metavar='CENTER_ALIGNED_COLUMNS', nargs='*',
                        type=int, default=[], help='column numbers with center alignment (from zero)')
    parser.add_argument('-r', '--right-aligned-columns', metavar='RIGHT_ALIGNED_COLUMNS', nargs='*',
                        type=int, default=[], help='column numbers with right alignment (from zero)')
    args = parser.parse_args()

    struct = json.load(sys.stdin)
    command = f"xargs -l bash -c '{args.command}'"
    cells = map_2d_dict(struct, command)
    table = Table(cells)
    print(table.markdown(args.center_aligned_columns, args.right_aligned_columns))
    return

if __name__=='__main__':
    main()
