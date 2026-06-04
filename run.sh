#!/bin/bash

DEPTH=${1:-6}
BRANCH_FACT=${2:-2}
DEPTH_VAR=${3:-1}
BRANCH_VAR=${4:-0}
RATE=${5:-0.2}

STEM="tree_d${DEPTH}_b${BRANCH_FACT}_dv${DEPTH_VAR}_bv${BRANCH_VAR}_$(date +%Y%m%d_%H%M%S)"
ORIGINAL_STEM="${STEM}_original"
CORRUPT_STEM="${ORIGINAL_STEM}_missing_$(python3 -c "print(int($RATE * 100))")"

echo "Running pipeline for: $STEM"

echo 
echo 
echo 
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### #####     TREE GENERATION     ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
python ./pipeline/tree_gen.py \
    -depth $DEPTH \
    -branch_fact $BRANCH_FACT \
    -depth_var $DEPTH_VAR \
    -branch_var $BRANCH_VAR \
    -name $STEM

echo 
echo 
echo 
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### #####     TREE CORRUPTION     ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
python ./pipeline/tree_corrupt.py $ORIGINAL_STEM -rate $RATE

echo 
echo 
echo 
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### #####     POINCARE EMBEDDING     ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
./train.sh $CORRUPT_STEM

echo 
echo 
echo 
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### #####     TREE RECONSTRUCTION     ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
python ./pipeline/tree_rec.py $CORRUPT_STEM

echo 
echo 
echo 
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "      Done. Look into artefacts in pipeline/artefacts/"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"
echo "##### ##### ##### ##### ##### ##### ##### ##### ##### #####"