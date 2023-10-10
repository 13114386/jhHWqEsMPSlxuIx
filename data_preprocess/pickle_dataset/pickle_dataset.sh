#!/bin/bash

echo "Usage: pickle_dataset.sh "
echo "        --dataset_root /parent_directory/to/an_built_dataset"
echo "        --dataset_folder the_built_dataset_folder"
echo "        --split_type train|validation|test"
echo "        --pair_type 'article,highlights'|'document,summary'"
echo "        --ext_type dataset.json"
echo "        --python_venv_dir /your/python/venv/bin/directory"
echo "Note: (argument order is important!)"
echo "Press enter Y/y to continue, N/n to abort"
read -r input


if [[ "$input" = "y" ]] || [[ "$input" = "Y" ]]; then
    PYTHON_VENV_BIN_DIR=${12}
    echo "Python venv bin dir: ${PYTHON_VENV_BIN_DIR}"
    source ${PYTHON_VENV_BIN_DIR}/activate
    export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/..:$PWD/../.."

    python3 pickle_dataset.py \
        --dataset_root   $2 \
        --dataset_folder $4 \
        --split_type     $6 \
        --pair_type      $8 \
        --ext_type       ${10} &
else
    echo "--dataset_root       $2"
    echo "--dataset_folder     $4"
    echo "--split_type         $6"
    echo "--pair_type          $8"
    echo "--ext_type           ${10}"
    echo "--python_venv_dir    ${12}"
    return
fi
