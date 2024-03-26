# a bash script to run pyserini bm25 on the beir datasets
# and convert the results to DPR format
# available datasets are trec-covid, bioasq, nfcorpus, hotpotqa
# fiqa, signal1m, trec-news, robust04, arguana, webis-touche2020
# quora, dbpedia-entity, scidocs, fever, climate-fever, scifact

export PYSERINI_CACHE="$(pwd)/cache/pyserini"

# define a list variable of all datasets
datasets=(trec-covid bioasq nfcorpus hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 quora dbpedia-entity scidocs fever climate-fever scifact)

# iterate over the datasets
for dataset in "${datasets[@]}"; do
    # run pyserini bm25 on the dataset
    python -m pyserini.search.lucene \
        --index beir-v1.0.0-${dataset}.flat \
        --topics beir-v1.0.0-${dataset}-test \
        --output outputs/bm25_flat/${dataset}.txt \
        --output-format trec \
        --batch 512 --threads 32 \
        --hits 100 --bm25 --remove-query
done

# run pyserini bm25 on the dl19-passage and dl20 datasets
datasets=(dl19-passage dl20)

for dataset in "${datasets[@]}"; do
    python -m pyserini.search.lucene \
        --index msmarco-v1-passage \
        --topics ${dataset} \
        --output outputs/bm25_flat/${dataset}.txt \
        --output-format trec \
        --batch 512 --threads 32 \
        --hits 100 --bm25 --remove-query
done


# convert BM25 results to DPR format
datasets=(trec-covid bioasq nfcorpus hotpotqa fiqa signal1m trec-news robust04 arguana webis-touche2020 quora dbpedia-entity scidocs fever climate-fever scifact)

for dataset in "${datasets[@]}"; do
    python3 convert_trec_to_dpr.py \
        outputs/bm25_flat/${dataset}.txt \
        beir-v1.0.0-${dataset}.flat \
        beir-v1.0.0-${dataset}-test \
        outputs/bm25_flat/${dataset}.json
done


# convert BM25 results to DPR format for dl19-passage and dl20
datasets=(dl19-passage dl20)

for dataset in "${datasets[@]}"; do
    python3 convert_trec_to_dpr.py \
        outputs/bm25_flat/${dataset}.txt \
        msmarco-v1-passage \
        ${dataset} \
        outputs/bm25_flat/${dataset}.json
done