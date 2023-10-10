#!/bin/bash

help()
{
    echo "Usage: . corenlp_annotate.sh "
    echo "              --port              [a_port_number(9000)]"
    echo "              --datasource        [cnndm|xsum]"
    echo "              --dataset_root      [/root/dataset/set/to/be/common/to/datasets/directories/or/folders]"
    echo "              --downloaded_folder [/folder/to/downloaded/dataset]"
    echo "              --split_type        [train|validation|test]"
    echo "              --column_names      [article,highlights]|[document,summary]"
    echo "              --mode              [PROP_DEFAULT|PROP_NER_COREF]"
    echo "              --output_dir        [a_parent_folder/to/output_folder|.]"
    echo "              --python_venv_dir   [/your/python/venv/bin/directory]"
}


NUM_ARGUMENTS=$#
EXPECTED_N_ARGS=18
if [ "$NUM_ARGUMENTS" -ne ${EXPECTED_N_ARGS} ]; then
    help
    return
fi


while :
do
  case "$1" in
    --port )
      PORT="$2"
      shift 2
      ;;
    --dataset_root )
      DATASET_ROOT="$2"
      shift 2
      ;;
    --datasource )
      DATASOURCE="$2"
      shift 2
      ;;
    --downloaded_folder )
      DOWNLOADED_FOLDER="$2"
      shift 2
      ;;
    --split_types )
      SPLIT_TYPES="$2"
      shift 2
      ;;
    --column_names )
      COLUMN_NAMES="$2"
      shift 2
      ;;
    --mode )
      MODE="$2"
      shift 2
      ;;
    --output_folder )
      OUTPUT_FOLDER="$2"
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


source ${PYTHON_VENV_BIN_DIR}/activate
export PYTHONPATH=$PYTHONPATH:$PWD:$PWD/..:$PWD/../..
export CLASSPATH=$CLASSPATH:${OUTPUT_FOLDER}/*:

RUN_TRACE_DIR="${DATASET_ROOT}/run_trace"
[ -d ${RUN_TRACE_DIR} ] || mkdir -p ${RUN_TRACE_DIR}

today=`date '+%Y_%m_%d_%H_%M'`;
LOGFILE="${RUN_TRACE_DIR}/${DATASOURCE}.corenlp.parse.${SPLIT_TYPE}.$today.err"
echo ${LOGFILE}
echo $HOSTNAME >${LOGFILE}

declare -A MODE_FOLDER_DICT
MODE_FOLDER_DICT["PROP_NER_COREF"]=".ner_coref"
MODE_FOLDER_DICT["PROP_DEFAULT"]=""

FOLDER_SUFFIX=${MODE_FOLDER_DICT["${MODE}"]}
OUTPUT_DIR=${OUTPUT_FOLDER}/corenlp.parse${FOLDER_SUFFIX}
echo "Output directory: ${OUTPUT_DIR}"

nohup python corenlp_annotate.py \
    --port          ${PORT} \
    --dataset_root  ${DATASET_ROOT} \
    --split_type    ${SPLIT_TYPE} \
    --column_names  ${COLUMN_NAMES}
    --corenlp_mode  ${MODE} \
    --source_folder ${DOWNLOAD_FOLDER} \
    --output_folder ${OUTPUT_DIR} 2>${LOGFILE} >/dev/null &

