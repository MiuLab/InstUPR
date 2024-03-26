import argparse
import tempfile
import json
import os

import numpy as np
import torch


QRELS = {
    "dl19-passage": "dl19-passage",
    "dl20": "dl20-passage",
    "trec-covid": "beir-v1.0.0-trec-covid-test",
    "bioasq": "beir-v1.0.0-bioasq-test",
    "nfcorpus": "beir-v1.0.0-nfcorpus-test",
    "hotpotqa": "beir-v1.0.0-hotpotqa-test",
    "fiqa": "beir-v1.0.0-fiqa-test",
    "signal1m": "beir-v1.0.0-signal1m-test",
    "trec-news": "beir-v1.0.0-trec-news-test",
    "robust04": "beir-v1.0.0-robust04-test",
    "arguana": "beir-v1.0.0-arguana-test",
    "webis-touche2020": "beir-v1.0.0-webis-touche2020-test",
    "quora": "beir-v1.0.0-quora-test",
    "dbpedia-entity": "beir-v1.0.0-dbpedia-entity-test",
    "scidocs": "beir-v1.0.0-scidocs-test",
    "fever": "beir-v1.0.0-fever-test",
    "climate-fever": "beir-v1.0.0-climate-fever-test",
    "scifact": "beir-v1.0.0-scifact-test",
}


def calculate_top_k_recall(top_k_results, string_prefix=""):
    n_docs = len(top_k_results[list(top_k_results.keys())[0]])
    print(string_prefix, f"n_docs: {n_docs}")
    for k, results in top_k_results.items():
        print(f"Top-{k}: {np.mean(results) * 100:.2f}")

    print()


def calculate_hard_score(ctx, rescore_type):
    if rescore_type == "answerable":
        return 1.0 if ctx['option_probs'][0] > ctx['option_probs'][1] else 0.0
    elif rescore_type == "1-5":
        return np.argmax(ctx['option_probs']) + 1


def calculate_soft_score(ctx, rescore_type, normalize_prob=False, softmax_prob=False):
    if normalize_prob:
        total_prob = sum(ctx['option_probs'])
        if total_prob < 1e-5:
            return None
        ctx['option_probs'] = [p / total_prob for p in ctx['option_probs']]
    elif softmax_prob:
        ctx['option_probs'] = torch.softmax(torch.tensor(ctx['option_logits']), dim=-1).tolist()

    if rescore_type == "answerable":
        return ctx['option_probs'][0]
    elif rescore_type == "1-5":
        return sum([(i + 1) * p for i, p in enumerate(ctx['option_probs'])])


def calculate_gen_score(ctx, rescore_type):
    lm_output = ctx['lm_output']
    if rescore_type == "answerable":
        if lm_output.strip().lower() == "yes":
            return 1.0
        elif lm_output.strip().lower() == "no":
            return 0.0
        else:
            return None
    elif rescore_type == "1-5":
        try:
            return float(lm_output.strip())
        except ValueError:
            return None


def rerank(outputs, rescore_type, normalize_prob=False, softmax_prob=False, argmax=False, reverse=False):
    illegal_count = 0
    for question in outputs:
        for ctx in question['ctxs']:
            if "lm_output" in ctx:
                score = calculate_gen_score(ctx, rescore_type)
            elif argmax:
                score = calculate_hard_score(ctx, rescore_type)
            elif "option_probs" in ctx or "option_logits" in ctx:
                score = calculate_soft_score(ctx, rescore_type, normalize_prob, softmax_prob)
            else:
                print("Illegal format. Reranking aborted.")
                exit(1)
            if score is None:
                illegal_count += 1
                score = 0.0

            ctx['score'] = score if not reverse else -score
        question['ctxs'] = sorted(question['ctxs'], key=lambda x: x['score'], reverse=True)

    return illegal_count


def evaluate_trec_eval(outputs, dataset):
    # Write to trec_eval format
    temp_file = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_file, 'w') as trec_file:
        for question in outputs:
            qid = question['id'] if 'id' in question else question['q_id']
            for i, ctx in enumerate(question['ctxs']):
                trec_file.write(f"{qid} Q0 {ctx['id']} {i + 1} {ctx['score']} rerank\n")

    from trec_eval import run_trec_eval
    run_trec_eval(['', '-c', '-m', 'ndcg_cut.10', QRELS[dataset], temp_file])
    run_trec_eval(['', '-c', '-m', 'recall.10', QRELS[dataset], temp_file])


def evaluate(outputs, per_language=False):
    top_k_results = {k: [] for k in [1, 5, 20, 100]}
    top_k_results_per_language = {}

    for question in outputs:
        rank = 10000
        for i, ctx in enumerate(question['ctxs']):
            if ctx['has_answer']:
                rank = i + 1
                break

        for k in top_k_results.keys():
            top_k_results[k].append(1 if rank <= k else 0)

        lang = question['lang']
        if lang not in top_k_results_per_language:
            top_k_results_per_language[lang] = {k: [] for k in [1, 5, 20, 100]}

        for k in top_k_results_per_language[lang].keys():
            top_k_results_per_language[lang][k].append(1 if rank <= k else 0)

    if per_language:
        for lang in top_k_results_per_language.keys():
            calculate_top_k_recall(
                top_k_results_per_language[lang],
                string_prefix=f"{lang} questions"
            )

    calculate_top_k_recall(top_k_results, string_prefix="All questions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="retriever output file")
    parser.add_argument("--output_file", default=None, help="If specified, will write the reranked output to this file")
    parser.add_argument("--rescore_type", default="1-5", choices=["1-5", "answerable"],
                        help="which type of rescoring to perform")
    parser.add_argument("--normalize", action='store_true', help="whether to normalize the probabilities")
    parser.add_argument("--softmax", action='store_true', help="whether to softmax the logits")
    parser.add_argument("--argmax", action='store_true', help="whether to argmax the logits")
    parser.add_argument("--trec_eval", action='store_true', help="whether to use trec_eval")
    parser.add_argument("--dataset", type=str, default=None, help="dataset name in case of trec_eval")
    parser.add_argument("--no_before", action='store_true', help="if enabled, will not show original performance")
    parser.add_argument("--per_language", action='store_true', help="whether show per language results")
    parser.add_argument("--reverse", action='store_true', help="whether to reverse the order of the results")
    args = parser.parse_args()

    if args.normalize and args.softmax:
        raise ValueError("You can only use one of --normalize and --softmax")

    if not os.path.exists(args.file):
        # for parsing purpose
        print("ndcg_cut_10\t\tFileNotFound")
        print("recall_100\t\tFileNotFound")
        exit(1)

    print("Loading retriever output file from", args.file)
    with open(args.file) as jsonfile:
        outputs = json.load(jsonfile)

    eval_type = "Generation" if outputs[0]['ctxs'][0].get('lm_output') else "Soft"
    print("-----------------------------------")
    print("Rescoring type:", args.rescore_type)
    print("Evaluation type:", eval_type)
    print("Normalize:", args.normalize)
    print("Softmax:", args.softmax)
    print("Argmax:", args.argmax)
    print("Trec eval:", args.trec_eval)
    print("# contexts:", len(outputs[0]['ctxs']))
    print("-----------------------------------")

    if not args.no_before:
        print("Before reranking")
        if args.trec_eval:
            evaluate_trec_eval(outputs, args.dataset)
        else:
            evaluate(outputs, args.per_language)

    illegal_count = rerank(outputs, args.rescore_type, args.normalize, args.softmax, args.argmax, args.reverse)

    print("After reranking")
    if args.trec_eval:
        evaluate_trec_eval(outputs, args.dataset)
    else:
        evaluate(outputs, args.per_language)

    if args.output_file is not None:
        print("Writing reranked output to", args.output_file)
        with open(args.output_file, 'w') as jsonfile:
            json.dump(outputs, jsonfile)

    print(f"Illegal count: {illegal_count}")
