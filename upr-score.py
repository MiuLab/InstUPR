import random
import numpy
import json
import time
import argparse
import os
import shutil

import torch
import torch.distributed as dist
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
from tqdm import tqdm

from utils import (
    print_rank_0,
    get_openqa_dataset,
    get_one_epoch_dataloader,
    initialize_distributed
)


LANG_ID_TO_LANG = {
    "ar": "Arabic",
    "bn": "Bengali",
    "fi": "Finnish",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "te": "Telugu",
    "en": "English",
    "es": "Spanish",
    "km": "Khmer",
    "ms": "Malay",
    "tr": "Turkish",
    "sv": "Swedish",
    "zh": "Chinese"
}


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


class UnsupervisedPassageReranker():
    def __init__(self, args):
        self.model = None
        self.dataloader = None
        self.dataset = None
        self.evidence_dataset = None

        self.args = args
        self.log_interval = args.log_interval
        # Hard coding the per gpu batch size to 1
        self.batch_size = 1

        self.load_attributes()
        self.is_main_builder = dist.get_rank() == 0
        self.num_total_builders = dist.get_world_size()

        self.temp_dir_name = args.reranker_output_dir

    def load_attributes(self):
        print_rank_0("Loading {} weights".format(self.args.hf_model_name))

        model_class = AutoModelForCausalLM if self.args.causal_model else AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.args.use_int8:
            self.model = model_class.from_pretrained(
                self.args.hf_model_name,
                device_map="auto",
                load_in_8bit=True
            )
        else:
            self.model = model_class.from_pretrained(
                self.args.hf_model_name,
                torch_dtype=torch.float16 if self.args.use_fp16 else torch.float32
            )

        for param in self.model.parameters():
            param.requires_grad = False

        if self.args.use_gpu and not self.args.use_int8:
            self.model = self.model.cuda()

        print_rank_0("Loaded {} weights".format(self.args.hf_model_name))

        # disable dropout
        self.model.eval()
        self.dataset = get_openqa_dataset(self.args.task_name,
                                          self.args.retriever_topk_passages_path,
                                          sample_rate=self.args.sample_rate)

        self.dataloader = iter(get_one_epoch_dataloader(self.dataset,
                                                        self.args,
                                                        self.batch_size))
        self.iteration = self.total_processed = 0

    def do_inference(self):
        reranked_answers_list = []
        original_answers_list = []
        reranked_data = []

        start_time = time.time()
        progress_bar = tqdm(
            total=self.args.max_inference_samples or len(self.dataset),
            disable=not self.is_main_builder
        )

        with open(self.args.instruction_file) as jsonfile:
            instruction = json.load(jsonfile)

        if not instruction["generate"]:
            assert isinstance(instruction["options"], list)
            option_ids = self.tokenizer.convert_tokens_to_ids(instruction["options"])

        encoder_max_length = instruction.get("encoder_max_length", 512)

        while True:
            try:
                # batch also has query_tokens and query_pad_data
                batch = next(self.dataloader)
            except (StopIteration, IndexError):
                break

            assert len(batch['id']) == 1, "Currently, we are doing inference with batch size 1"

            all_contexts = batch['encoder_ids'][0][:self.args.topk_passages]

            all_ids = []
            has_answer_list = []

            for i, context in enumerate(all_contexts):
                text = context["text"]
                title = context["title"]
                question = batch['question'][0]

                context_text = f"{title} {text}"
                if instruction.get("context_max_length", None) is not None:
                    context_tokens = self.tokenizer(context_text).input_ids
                    context_tokens = context_tokens[:instruction["context_max_length"]]
                    context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=True)

                if instruction.get("question_max_length", None) is not None:
                    question_tokens = self.tokenizer(question).input_ids
                    if instruction.get("question_max_length_right", None) is not None:
                        question_tokens = question_tokens[:instruction["question_max_length"]] + \
                            question_tokens[-instruction["question_max_length_right"]:]
                    else:
                        question_tokens = question_tokens[:instruction["question_max_length"]]
                    question = self.tokenizer.decode(question_tokens, skip_special_tokens=True)

                if "{examples}" in instruction["instruction"]:
                    example_template = instruction["example_template"]
                    formatted_examples = []
                    for example in instruction["examples"]:
                        formatted_examples.append(example_template.format(**example))

                    input_text = instruction["instruction"].format(
                        context=context_text,
                        question=question,
                        examples="\n\n".join(formatted_examples)
                    )
                else:
                    input_text = instruction["instruction"].format(
                        context=context_text, question=question
                    )
                all_ids.append(input_text)
                has_answer_list.append(context.get('has_answer'))

            input_encoding = self.tokenizer(all_ids,
                                            padding='longest',
                                            max_length=encoder_max_length,
                                            pad_to_multiple_of=8,
                                            truncation=True,
                                            return_tensors='pt')

            context_tensor, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
            decoder_input_ids = torch.zeros((len(context_tensor), 1)).long()
            if self.args.use_gpu:
                context_tensor = context_tensor.cuda()
                attention_mask = attention_mask.cuda()
                decoder_input_ids = decoder_input_ids.cuda()

            all_outputs = []

            for i in range(0, len(context_tensor), self.args.shard_size):
                encoder_tensor_view = context_tensor[i: i + self.args.shard_size]
                attention_mask_view = attention_mask[i: i + self.args.shard_size]
                decoder_input_ids_view = decoder_input_ids[i: i + self.args.shard_size]

                with torch.no_grad():
                    if instruction["generate"]:
                        outputs = self.model.generate(encoder_tensor_view, max_new_tokens=8)
                    else:
                        max_length_shard = attention_mask_view.sum(dim=1).max().item()
                        if self.tokenizer.padding_side == "left":
                            encoder_tensor_view = encoder_tensor_view[:, -max_length_shard:]
                            attention_mask_view = attention_mask_view[:, -max_length_shard:]
                        else:
                            encoder_tensor_view = encoder_tensor_view[:, :max_length_shard]
                            attention_mask_view = attention_mask_view[:, :max_length_shard]

                        if self.args.causal_model:
                            outputs = self.model(
                                encoder_tensor_view,
                                attention_mask=attention_mask_view
                            ).logits
                        else:
                            outputs = self.model(
                                encoder_tensor_view,
                                attention_mask=attention_mask_view,
                                decoder_input_ids=decoder_input_ids_view
                            ).logits

                all_outputs.extend(outputs)

            for i, ctx in enumerate(all_contexts):
                if instruction["generate"]:
                    ctx["lm_output"] = self.tokenizer.decode(all_outputs[i], skip_special_tokens=True)
                else:
                    ctx["options"] = instruction["options"]
                    option_logits = [all_outputs[i][-1][j].detach().float().cpu().item() for j in option_ids]
                    ctx["option_logits"] = option_logits
                    probs = torch.softmax(all_outputs[i][-1], dim=-1)
                    ctx["option_probs"] = [probs[j].detach().float().cpu().item() for j in option_ids]

            item = {
                "id": batch["id"][0],
                "question": batch['question'][0],
                "answers": batch['answers'][0],
                "ctxs": all_contexts,
                "lang": batch['lang'][0]
            }
            reranked_data.append(item)

            if self.args.max_inference_samples and len(reranked_data) >= self.args.max_inference_samples:
                break

            progress_bar.update(len(batch["id"]))

        end_time = time.time()
        time_taken = end_time - start_time
        time_taken_per_question = time_taken / len(reranked_data)
        torch.distributed.barrier()

        print_rank_0("Time taken: {} seconds".format(time_taken))
        print_rank_0("Time taken per question: {} seconds".format(time_taken_per_question))
        self.save_shard(reranked_data)

        answers_per_language = {}
        for item, orig, reranked in zip(reranked_data, original_answers_list, reranked_answers_list):
            lang = item['lang']
            if lang not in answers_per_language:
                answers_per_language[lang] = ([], [])
            answers_per_language[lang][0].append(orig)
            answers_per_language[lang][1].append(reranked)

        for language in answers_per_language.keys():
            self.compute_topk_recall(
                answers_per_language[language][0],
                string_prefix=f"Original Ranking - {language}"
            )
            self.compute_topk_recall(
                answers_per_language[language][1],
                string_prefix=f"Re-Ranking - {language}"
            )

        self.compute_topk_recall(original_answers_list, string_prefix="Original Ranking")
        self.compute_topk_recall(reranked_answers_list, string_prefix="Re-Ranking")

        if self.args.merge_shards_and_save:
            self.save_shard(reranked_data)

        del self.model
        # This process signals to finalize its shard and then synchronize with the other processes
        torch.distributed.barrier()

        if self.args.merge_shards_and_save:
            # rank 0 process builds the final copy
            if self.is_main_builder:
                self.merge_shards_and_save()
            # complete building the final copy
            torch.distributed.barrier()

    @staticmethod
    def calculate_topk_hits(scores, max_k):
        top_k_hits = [0] * max_k
        for question_hits in scores:
            best_hit = next((i for i, x in enumerate(question_hits[:max_k]) if x), None)
            if best_hit is not None:
                top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
        return top_k_hits

    def compute_topk_recall(self, answers_list, string_prefix):
        topk_hits = self.calculate_topk_hits(answers_list, max_k=self.args.report_topk_accuracies[-1])

        topk_hits = torch.FloatTensor(topk_hits).cuda()
        n_docs = torch.FloatTensor([len(answers_list)]).cuda()
        torch.distributed.all_reduce(topk_hits, torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(n_docs, torch.distributed.ReduceOp.SUM)

        if torch.distributed.get_rank() == 0:
            topk_hits = topk_hits / n_docs
            print(f"{string_prefix}. # docs: {n_docs.item()}")
            for i in self.args.report_topk_accuracies:
                print_rank_0("top-{}: {:.2f}".format(i, topk_hits[i - 1] * 100))
            print()

    def save_shard(self, data):
        """
        Save the block data that was created this in this process
        """
        if not os.path.isdir(self.temp_dir_name):
            os.makedirs(self.temp_dir_name, exist_ok=True)

        outpath = os.path.join(self.temp_dir_name, "rank{}.json".format(dist.get_rank()))
        with open(outpath, "w") as writer:
            writer.write(json.dumps(data, indent=4) + "\n",)
        print(f"Wrote output to {outpath}")

    def merge_shards_and_save(self):
        """Combine all the shards made using self.save_shard()"""
        shard_names = os.listdir(self.temp_dir_name)
        all_data = []

        for fname in os.listdir(self.temp_dir_name):
            shard_size = 0
            old_size = len(all_data)
            fpath = '{}/{}'.format(self.temp_dir_name, fname)
            with open(fpath, 'r') as f:
                data = json.load(f)
                shard_size = len(data)
                all_data.extend(data)

            assert len(all_data) == old_size + shard_size
            os.remove(fpath)

        # save the consolidated shards
        outpath = os.path.join(self.args.reranker_output_dir, "{}.json".format(self.args.special_suffix))

        with open(outpath, 'w') as writer:
            writer.write(json.dumps(all_data, indent=4) + "\n")

        print("Finished merging {} shards for a total of {} embeds".format(
            len(shard_names), len(all_data)), flush=True)

        # make sure that every single piece of data was embedded
        assert len(all_data) == len(self.dataset)

        shutil.rmtree(self.temp_dir_name, ignore_errors=True)


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title='argument-parser')

    group.add_argument('--local_rank', type=int, default=-1,
                       help='local rank passed from distributed launcher.')

    group.add_argument('--main-port', type=int, default=29500,
                       help='Main port number.')

    group.add_argument('--special-suffix', type=str, default="",
                       help='special suffix extension for saving merged file')

    group.add_argument(
        '--retriever-topk-passages-path',
        type=str,
        default="downloads/data/retriever-outputs/nq-dev.json",
        help='Path of the Top-K passage output file from retriever (.json file)'
    )

    group.add_argument('--topk-passages', type=int, default=1000,
                       help='number of topk passages to select')

    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')

    group.add_argument('--shard-size', type=int, default=16)

    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")

    group.add_argument('--reranker-output-dir', type=str, default="downloads/data/retriever-outputs/",
                       help='Path to save UPR results')

    group.add_argument('--task-name', type=str, default="reranking",
                       help='Name of the task.')

    group.add_argument('--hf-model-name', type=str, default="t5-large",
                       help='Name of the HF model.')

    group.add_argument('--interactive-node', action='store_true',
                       help='If the node is interactive or not')

    group.add_argument('--use-gpu', action='store_true',
                       help='Use GPU or not')

    group.add_argument('--use-fp16', action='store_true',
                       help='Whether to use FP16 data format for the T0/T5 models')

    group.add_argument('--use-int8', action='store_true',
                       help='Whether to use INT8 data format for the T0/T5 models')

    group.add_argument('--causal-model', action='store_true',
                       help='Whether this model is a causal model such as LLaMA')

    group.add_argument('--merge-shards-and-save', action='store_true',
                       help='whether to merge individual data shards or not for reranking')

    group.add_argument('--sample-rate', type=float, default=1.,
                       help="Sample rate for the number of examples.")

    group.add_argument('--max-inference-samples', type=int, default=None,
                       help="Maximum number of examples to perform inference.")

    group.add_argument('--random-seed', type=int, default=1234,
                       help="Random seed.")

    group.add_argument('--evidence-data-path', type=str, default=None,
                       help='Path to Wikipedia evidence passages file')

    group.add_argument('--instruction_file', type=str, default="instructions/default.json",
                       help='Instruction file for the model')

    group.add_argument('--report-topk-accuracies', nargs='+', type=int, default=[1, 5, 10, 20, 50, 100],
                       help="Which top-k accuracies to report (e.g. '1 5 20')")

    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    return args


def main():
    args = get_args()
    set_random_seed(args.random_seed)
    initialize_distributed(args)

    reranker = UnsupervisedPassageReranker(args)
    reranker.do_inference()


if __name__ == "__main__":
    main()
