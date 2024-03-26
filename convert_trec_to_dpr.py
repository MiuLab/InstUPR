# Convert output from TREC format to DPR format
# TREC format: query_id, query_name, doc_id, rank, score, run_name
# DPR format: [
#   {
#     "q_id": {query_id},
#     "question": {query},
#     "answers": [
#       ""
#     ],
#     "lang": "en",
#     "ctxs": [
#       {
#           "id": {doc_id},
#           "title": {title},
#           "text": {text},
#           "score": {score},
#           "has_answer": false
#       }
#     ]
#   }
# ]

import argparse
import json

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics


def convert(input_file, index, topics_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    searcher = LuceneSearcher.from_prebuilt_index(index)
    topics = get_topics(topics_file)

    data = {}
    for line in tqdm(lines):
        query_id, query, doc_id, rank, score, run_name = line.split()
        doc = searcher.doc(doc_id)
        try:
            qa = topics[query_id]
        except KeyError:
            qa = topics[int(query_id)]

        if query_id not in data:
            data[query_id] = {
                "q_id": query_id,
                "question": qa["title"],
                "answers": [""],
                "lang": "en",
                "ctxs": [],
            }
        raw_doc = json.loads(doc.raw())
        if "title" not in raw_doc:
            raw_doc["title"] = ""
        if "text" not in raw_doc and "contents" in raw_doc:
            raw_doc["text"] = raw_doc["contents"]
        data[query_id]["ctxs"].append(
            {
                "id": doc_id,
                "title": raw_doc["title"],
                "text": raw_doc["text"],
                "score": float(score),
                "has_answer": False,
            }
        )

    with open(output_file, "w") as f:
        json.dump(list(data.values()), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("index", type=str)
    parser.add_argument("topics", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    convert(args.input_file, args.index, args.topics, args.output_file)