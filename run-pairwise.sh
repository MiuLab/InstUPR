export TRANSFORMERS_CACHE="$(pwd)/cache"

DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"

datasets=(trec-covid bioasq nfcorpus fiqa signal1m trec-news robust04 arguana webis-touche2020 dbpedia-entity scidocs climate-fever scifact dl19-passage dl20)

for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 ${DISTRIBUTED_ARGS} upr-pair.py \
      --num-workers 2 \
      --shard-size 8 \
      --topk-passages 30 \
      --hf-model-name "google/flan-t5-xl" \
      --use-gpu \
      --use-fp16 \
      --report-topk-accuracies 1 5 20 100 \
      --reranker-output-dir outputs/pairwise/${dataset}_flan-t5-xl \
      --retriever-topk-passages-path outputs/pointwise/${dataset}_flan-t5-xl/reranked.json \
      --instruction_file instructions/pairwise/${dataset}.json
done
