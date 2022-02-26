from argparse import ArgumentParser
from NLPDatasetIO.document import Document, Entity
from NLPDatasetIO.dataset import Dataset
import pandas as pd
import json


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--do_lower', action='store_true')
    return parser.parse_args()


def row_to_doc(row):
    entities = {}
    for idx, entity in enumerate(row['entity']):
        entity_id = f'T{idx}'
        entities[entity_id] = Entity(entity_id=entity_id, text=entity['text'],
                                     start=entity['start'], end=entity['end'], type='ADR')

    return Document(doc_id=row['tweet_id'], text=row['text'], entities=entities)


def read_input_data(fpath):
    input_data = pd.read_csv(fpath, sep='\t', encoding='utf-8')
    if args.do_lower:
        input_data['text'] = input_data.text.str.lower()
    input_data['entity'] = input_data.apply(lambda row: {
                                                            'start': row['start'],
                                                            'end': row['end'],
                                                            'text': row['span']
                                                        }, axis=1)
    input_data = input_data.groupby(['tweet_id', 'text'], sort=False)['entity'].apply(list).reset_index()
    documents = []
    for row_idx, row in input_data.iterrows():
        document = row_to_doc(row)
        documents.append(document)
    return Dataset(documents=documents)


def dataset_to_dict(dataset):
    dataset.detailed = False
    dicts = []
    for tokens, labels in dataset.iterate_token_level():
        dicts.append({'words': tokens, 'ner': labels})
    return dicts


def main(args):
    dataset = read_input_data(args.input)
    dicts = dataset_to_dict(dataset)
    with open(args.output, 'w', encoding='utf-8') as output_stream:
        for dict in dicts:
            serialized_dict = json.dumps(dict, ensure_ascii=False)
            output_stream.write(serialized_dict + '\n')


if __name__ == '__main__':
    args = get_args()
    main(args)