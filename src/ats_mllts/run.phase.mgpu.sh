#!/bin/bash


help()
{
    echo "Usage: run.phase.mgpu.sh --run_session_type       [train|test]"
    echo "                         --datasource             [cnn|xsum]"
    echo "                         --alias                  [cnndm|xsum]"
    echo "                         --dataset_root           [/root/directory/to/runtime/dataset/folders]"
    echo "                         --token_type             ['']"
    echo "                         --data_build_type        [struct|struct.ner_coref]"
    echo "                         --pretrained_model       [facebook/bart-base]"
    echo "                         --split_type             [\"train\",\"validation\"]"
    echo "                         --pair_type              [\"document\",\"summary\"]|[\"article\",\"highlights\"]"
    echo "                         --data_file_type         [doc|struct]"
    echo "                         --sd_vocab_folder        [struct|struct.ner_coref]"
    echo "                         --coref_vocab_folder     [struct|struct.ner_coref]"
    echo "                         --struct_feature_level   [none|struct|coref]"
    echo "                         --dataset_changed        [true|false]"
    echo "                         --query_model_size       [true|false]"
    echo "                         --python_venv_dir        [/your/python/venv/bin/directory]"
}

NUM_ARGUMENTS=$#
EXPECTED_N_ARGS=32
if [ "$NUM_ARGUMENTS" -ne ${EXPECTED_N_ARGS} ]; then
    help
    return
fi

while :
do
  case "$1" in
    --run_session_type )
      RUN_SESSION_TYPE="$2"
      shift 2
      ;;
    --datasource )
      DATASOURCE="$2"
      shift 2
      ;;
    --dataset_root )
      DATASET_ROOT="$2"
      shift 2
      ;;
    --alias )
      DATASOURCE_ALIAS="$2"
      shift 2
      ;;
    --token_type )
      TOKEN_TYPE="$2"
      shift 2
      ;;
    --data_build_type )
      DATASET_BUILD_TYPE="$2"
      shift 2
      ;;
    --pretrained_model )
      PRETRAINED_MODEL_TYPE="$2"
      shift 2
      ;;
    --split_type )
      DATA_SPLIT_TYPE="$2"
      shift 2
      ;;
    --pair_type )
      DATA_PAIR_TYPE="$2"
      shift 2
      ;;
    --data_file_type )
      DATA_FILE_TYPE="$2"
      shift 2
      ;;
    --struct_feature_level )
      STRUCT_FEATURE_LEVEL="$2"
      shift 2
      ;;
    --sd_vocab_folder )
      SD_VOCAB_FOLDER="$2"
      shift 2
      ;;
    --coref_vocab_folder )
      COREF_VOCAB_FOLDER="$2"
      shift 2
      ;;
    --query_model_size )
      QUERY_MODEL_SIZE="$2"
      shift 2
      ;;
    --dataset_changed )
      DATASET_CHANGED="$2"
      shift 2
      ;;
    --python_venv_dir )
      PYTHON_VENV_BIN_DIR="$2"
      shift 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      # echo "Unexpected option: $1"
      # help
      break
      ;;
  esac
done


echo "Python venv bin dir: ${PYTHON_VENV_BIN_DIR}"
source ${PYTHON_VENV_BIN_DIR}/activate
export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/.."
export PATH=/usr/local/cuda-11.3/bin:$PATH
export CPATH=/usr/local/cuda-11.3/include:$CPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

FOLDER_NAME="`basename $PWD`"
MATE_DIR=./


COREF_VOCAB_FOLDER=../${COREF_VOCAB_FOLDER}


DATA_DOC=""
if [[ ${DATA_FILE_TYPE} == "doc" ]]; then
    DATA_DOC=".doc"
fi

DATASET_CHANGED=[ ${DATASET_CHANGED} == "true" ]

QUERY_MODEL_SIZE=[ ${QUERY_MODEL_SIZE} == "true" ]

RUN_TRACE_DIR="${MATE_DIR}/${FOLDER_NAME}/run_trace"
[ -d ${RUN_TRACE_DIR} ] || mkdir -p ${RUN_TRACE_DIR}

today=`date '+%Y_%m_%d_%H_%M'`;
RUN_LOG="${RUN_TRACE_DIR}/${DATASOURCE_ALIAS}_${RUN_SESSION_TYPE}_results_$today.out"

echo ${RUN_LOG}
echo $HOSTNAME >${RUN_LOG}

echo "--modeldata_root:             ${MATE_DIR}/${FOLDER_NAME}"
echo "--dataset_root:               ${DATASET_ROOT}"
echo "--config_folder:              ${DATASOURCE_ALIAS}"
echo "--dataset_folder:             ${DATASET_BUILD_TYPE}"
echo "--base_model_pretrained_name: ${PRETRAINED_MODEL_TYPE}"
echo "--tokenizer_name:             ${PRETRAINED_MODEL_TYPE}"
echo "--split_type:                 ${DATA_SPLIT_TYPE}"
echo "--pair_type:                  ${DATA_PAIR_TYPE}"
echo "--dataset_file:               {split_type}.{pair_type}${DATA_DOC}.dataset.json"
echo "--struct_feature_level:       ${STRUCT_FEATURE_LEVEL}"
echo "--inarc_vocab_file:           ${SD_VOCAB_FOLDER}/inarc.vocab.json"
echo "--pos_vocab_file:             ${SD_VOCAB_FOLDER}/pos.vocab.json"
echo "--coref_animacy_vocab_file:   ${COREF_VOCAB_FOLDER}/coref.animacy.vocab.json"
echo "--coref_gender_vocab_file:    ${COREF_VOCAB_FOLDER}/coref.gender.vocab.json"
echo "--coref_number_vocab_file:    ${COREF_VOCAB_FOLDER}/coref.number.vocab.json"
echo "--coref_type_vocab_file:      ${COREF_VOCAB_FOLDER}/coref.type.vocab.json"
echo "--dataset_changed:            ${DATASET_CHANGED}"

#nohup python3 -u train_main.py
if [ "${RUN_SESSION_TYPE}" = "train" ]; then
    accelerate launch --config_file ./accelerate_config.${DATASOURCE_ALIAS}.yaml train_main.py \
        --modeldata_root             ${MATE_DIR}/${FOLDER_NAME} \
        --dataset_root               ${DATASET_ROOT} \
        --config_folder              ${DATASOURCE_ALIAS} \
        --dataset_folder             ${DATASET_BUILD_TYPE} \
        --base_model_pretrained_name ${PRETRAINED_MODEL_TYPE} \
        --tokenizer_name             ${PRETRAINED_MODEL_TYPE} \
        --use_slow_tokenizer                                  \
        --split_type                 ${DATA_SPLIT_TYPE} \
        --pair_type                  ${DATA_PAIR_TYPE} \
        --dataset_file               {split_type}.{pair_type}${DATA_DOC}.dataset.json \
        --struct_feature_level       ${STRUCT_FEATURE_LEVEL} \
        --modeling_choice            model_mrl \
        --inarc_vocab_file           ${SD_VOCAB_FOLDER}/inarc.vocab.json \
        --pos_vocab_file             ${SD_VOCAB_FOLDER}/pos.vocab.json \
        --coref_animacy_vocab_file   ${COREF_VOCAB_FOLDER}/coref.animacy.vocab.json \
        --coref_gender_vocab_file    ${COREF_VOCAB_FOLDER}/coref.gender.vocab.json \
        --coref_number_vocab_file    ${COREF_VOCAB_FOLDER}/coref.number.vocab.json \
        --coref_type_vocab_file      ${COREF_VOCAB_FOLDER}/coref.type.vocab.json \
        --seed                       19786403 \
        --time_limit                 -1 \
        --dataset_changed            ${DATASET_CHANGED} \
        --query_model_size           ${QUERY_MODEL_SIZE} \
        --skip_except                \
        --early_stop_count_on_rouge  3 >>${RUN_LOG} 2>&1 &
else
    accelerate launch --config_file ./accelerate_config.${DATASOURCE_ALIAS}.yaml test_main.py \
        --modeldata_root             ${MATE_DIR}/${FOLDER_NAME} \
        --dataset_root               ${DATASET_ROOT} \
        --config_folder              ${DATASOURCE_ALIAS} \
        --dataset_folder             ${DATASET_BUILD_TYPE} \
        --base_model_pretrained_name ${PRETRAINED_MODEL_TYPE} \
        --tokenizer_name             ${PRETRAINED_MODEL_TYPE} \
        --use_slow_tokenizer                                  \
        --split_type                 ${DATA_SPLIT_TYPE} \
        --pair_type                  ${DATA_PAIR_TYPE} \
        --dataset_file               {split_type}.{pair_type}${DATA_DOC}.dataset.json \
        --modeling_choice            model_mrl \
        --inarc_vocab_file           ${SD_VOCAB_FOLDER}/inarc.vocab.json \
        --pos_vocab_file             ${SD_VOCAB_FOLDER}/pos.vocab.json \
        --coref_animacy_vocab_file   ${COREF_VOCAB_FOLDER}/coref.animacy.vocab.json \
        --coref_gender_vocab_file    ${COREF_VOCAB_FOLDER}/coref.gender.vocab.json \
        --coref_number_vocab_file    ${COREF_VOCAB_FOLDER}/coref.number.vocab.json \
        --coref_type_vocab_file      ${COREF_VOCAB_FOLDER}/coref.type.vocab.json \
        --test_batch_size            4 >>${RUN_LOG} 2>&1 &
fi
#fi
