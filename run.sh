#!/bin/sh

NAME=$1
CORR_TYPE=$2
SEED=$3

echo "==> Generating tree..."
python pipeline/tree_gen.py \
    -name ${NAME} \
    -depth 6 \
    -branch_fact 2 \
    -stop_prob 0.0 \
    -chain_prob 0.0 \
    -seed $SEED

echo "==> Corrupting closure..."
python pipeline/tree_corrupt.py ${NAME} \
    -type ${CORR_TYPE} \
    -rate 0.1 \
    -seed $SEED

echo "==> Training Poincare embeddings..."
python embed.py \
    -dim 5 \
    -lr 0.3 \
    -epochs 50 \
    -negs 50 \
    -burnin 10 \
    -ndproc 1 \
    -model distance \
    -manifold poincare \
    -dset pipeline/artefacts/${NAME}_${CORR_TYPE}.csv \
    -checkpoint pipeline/artefacts/${NAME}_${CORR_TYPE}.pth \
    -batchsize 10 \
    -eval_each 1 \
    -fresh \
    -sparse \
    -train_threads 1 \
    -gpu -1

echo "==> Recovering tree..."
python pipeline/tree_rec.py ${NAME}_${CORR_TYPE} \
    -method poincare
python pipeline/tree_rec.py ${NAME}_${CORR_TYPE} \
    -method angular

echo "==> Evaluating corrupted closure as is..."
python pipeline/closure_comp.py ${NAME} ${NAME}_${CORR_TYPE}

echo "==> Evaluating recovered closure with reverse Poincare algorithm..."
python pipeline/closure_comp.py ${NAME} ${NAME}_${CORR_TYPE}_recovered_poincare

echo "==> Evaluating recovered closure with respect to angle..."
python pipeline/closure_comp.py ${NAME} ${NAME}_${CORR_TYPE}_recovered_angular