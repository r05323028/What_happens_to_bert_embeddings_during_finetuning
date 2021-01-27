import click
import pickle
from pathlib import Path

import tensorflow_datasets as tfdf


@click.group()
def cli():
    """
    Bert Repro Command Line Interface
    """


@click.command()
@click.option('--dataset',
              type=click.Choice(['squad', 'glue/mnli']),
              required=True)
@click.argument('output_dir', type=Path, default="./data")
def download(dataset, output_dir):
    '''
    Download dataset
    '''
    tfdf.load(dataset, data_dir=output_dir)


cli.add_command(download)

if __name__ == '__main__':
    cli()