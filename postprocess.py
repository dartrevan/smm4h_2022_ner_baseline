from argparse import ArgumentParser
from NLPDatasetIO.data_io.utils import extract_entities
import pandas as pd
import json


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--predicted_labels')
    parser.add_argument('--tokens')
    parser.add_argument('--documents')
    parser.add_argument('--save_to')
    return parser.parse_args()


def read_labels(fpath):
    with open(fpath, encoding='utf-8') as input_stream:
        labels = [line.strip().split() for line in input_stream]
    return labels


def read_tokens(fpath):
    with open(fpath, encoding='utf-8') as input_stream:
        tokens = [json.loads(line) for line in input_stream]
    return tokens


def get_entities(documents, tokens, labels):
    all_entities = []
    tokens = [token['words'] for token in tokens]
    for document, document_tokens, document_labels in zip(documents, tokens, labels):
        document_entities, _ = extract_entities(document_tokens, document_labels, document['text'].lower())
        for entity in document_entities.values():
            all_entities.append({
                'tweet_id': document['tweet_id'],
                'start': entity.start,
                'end': entity.end,
                'span': document['text'][entity.start:entity.end]
            })
    return pd.DataFrame(all_entities)


def main(args):
    predicted_labels = read_labels(args.predicted_labels)
    tokens = read_tokens(args.tokens)
    documents = pd.read_csv(args.documents, sep='\t', encoding='utf-8')
    documents = documents[['tweet_id', 'user_id', 'created_at', 'text']].drop_duplicates()
    entities = get_entities(documents.to_dict('records'), tokens, predicted_labels)
    entities = pd.merge(documents, entities, on='tweet_id', how='left')
    entities.to_csv(args.save_to, index=False, sep='\t', encoding='utf-8')


if __name__ == '__main__':
    args = get_args()
    main(args)