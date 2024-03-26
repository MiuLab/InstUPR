# InstUPR : Instruction-based Unsupervised Passage Reranking with Large Language Models

[[Paper]](https://arxiv.org/abs/2403.16435)

This repository contains the source code of our paper "InstUPR: Instruction-based Unsupervised Passage Reranking with Large Language Models".

## Requirements
* Python >= 3.8
* transformers
* torch
* pyserini
* trec_eval

Please install all required packages listed in `requirements.txt` by running the following command:
```bash
pip install -r requirements.txt
```

## Data
We use the [BEIR](https://github.com/beir-cellar/beir) benchmark and the TREC DL19 and DL20 datasets in our experiments. Since we utilize the BM25 retrieval results from [Pyserini](https://github.com/castorini/pyserini) for reranking, we recommend using Pyserini's 2CR commands to perform BM25 retrieval from their pre-built index. We provide a script to run BM25 retrieval on all datasets and convert them to our desired format (DPR format):
```bash
bash run_pyserini_bm25.sh
```

You can also download the preprocessed data from our [Google Drive (7.7GB)](https://drive.google.com/file/d/1u82Pw9fFhXw0IaBs_SL8IR-JpSF1E77n/view?usp=sharing).


## Run InstUPR

### Pointwise Reranking
After running BM25 retrieval, you can perform pointwise reranking on the top-100 passages using InstUPR with the following command:
```bash
bash run-pointwise.sh
```
This script will perform pointwise reranking on the top-100 passages for each query in the BEIR benchmark and the TREC DL19 and DL20 datasets.

#### Evaluation
To evaluate the reranking results, you need to install the `trec_eval` tool. We provide a script to evaluate the reranking results:
```bash
python3 rerank_score.py \
    outputs/pointwise/dl19-passage_flan-t5-xl/rank0.json \
    --trec_eval \
    --dataset dl19-passage \
    --softmax \
    --output_file outputs/pointwise/dl19-passage_flan-t5-xl/reranked.json
```
This command will evaluate the reranking results and save the reranked results in the output file.

### Pairwise Reranking
After running pointwise reranking, you can perform pairwise reranking to the reranked passages using InstUPR with the following command:
```bash
bash run-pairwise.sh
```
This script will perform pairwise reranking on the reranked passages for each query in the BEIR benchmark and the TREC DL19 and DL20 datasets. Note that this script reranks the top-30 passages for each query. You can adjust this number by changing the `--topk-passages` argument in the script.

#### Evaluation
Evaluation is the same as the pointwise reranking evaluation.


## Reference
If you find our work useful, please consider citing our paper:
```
@misc{huang2024instupr,
      title={InstUPR : Instruction-based Unsupervised Passage Reranking with Large Language Models}, 
      author={Chao-Wei Huang and Yun-Nung Chen},
      year={2024},
      eprint={2403.16435},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```