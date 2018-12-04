from __future__ import absolute_import

import logging
import argparse

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions



class Split(beam.DoFn):
    '''Splits each row on commas and returns dictionary representing the row'''

    def process(self, element):
        country, duration, user = element.split(',')
        return [{
            'country': country,
            'duration': float(duration),
            'user': user
        }]


class CollectTimings(beam.DoFn):
    '''Returns a list of tuples containing country and duration'''

    def process(self, element):
        return [(element['country'], element['duration'])]


class CollectUsers(beam.DoFn):
    '''Returns list of tuples containing country and user'''

    def process(self, element):
        return [(element['country'], element['user'])]


class FormatAsCSV(beam.DoFn):
    '''Formats the elements as a CSV row string for writting to Text'''

    def process(self, element):
        result = [
            "{},{},{}".format(
                element[0],
                element[1]['users'][0],
                element[1]['timings'][0]
            )
        ]
        return result


def run(argv=None):
    '''Main entry point; defines the webapp metrics pipeline.'''
    
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        dest='input',
                        default='dataflow/sample/input.csv',
                        help='Input file to process.')
    parser.add_argument('--output',
                        dest='output',
                        default='dataflow/sample/output.csv',
                        help='Output file to write results to.')
    
    known_args, pipeline_args = parser.parse_known_args(argv)

    # Pipeline options
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = 1 == 1
    p = beam.Pipeline(options=pipeline_options)

    rows = (
        p |
        'read input' >> ReadFromText(known_args.input) |
        'split elements' >> beam.ParDo(Split())
    )

    timings = (
        rows |
        beam.ParDo(CollectTimings()) |
        'Grouping Timings' >> beam.GroupByKey() |
        'Calculating Averages' >> beam.CombineValues(
            beam.combiners.MeanCombineFn()
        )
    )

    users = (
        rows |
        beam.ParDo(CollectUsers()) |
        'Grouping Users' >> beam.GroupByKey() |
        'Counting Users' >> beam.CombineValues(
            beam.combiners.CountCombineFn()
        )
    )

    to_be_joined = (
        {
            'timings': timings,
            'users': users
        } |
        beam.CoGroupByKey() |
        beam.ParDo(FormatAsCSV()) |
        WriteToText(known_args.output.split('.')[0], '.' + known_args.output.split('.')[-1])
    )

    result = p.run()
    result.wait_until_finish()

    print('Done!')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
