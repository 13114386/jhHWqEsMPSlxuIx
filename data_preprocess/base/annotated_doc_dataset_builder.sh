#!/bin/bash


help()
{
    echo "Usage: annotated_doc_dataset_builder.sh "
    echo "          --dataset_root        [/root/dataset/set/to/be/common/to/datasets/directories/or/folders]"
    echo "          --datasource          [cnndm|xsum]"
    echo "          --downloaded_folder   [directory/to/downloaded/dataset]"
    echo "          --source_folder       [directory/to/stanford_corenlp_annotated_output_files]"
    echo "          --output_folder       [folder/to/output_files]"
    echo "          --split_types         [train]|[validation]|[test]"
    echo "          --pair_types          [article,highlights]|[document,summary]"
    echo "          --tokenizer_name      [facebook/bart-base]"
    echo "          --annotated_data_dir  [an_optinal_parent_folder/to/above_output_folder|.]"
    echo "          --python_venv_dir     [/your/python/venv/bin/directory]"
}


#LONG=source_folder:,output_folder:,split_types:,build_vocab:,use_corefs:,analyse_corefs:,build_compose:,build_stype2id:
#OPTS=$(getopt --longoptions $LONG -- "$@")

NUM_ARGUMENTS=$#
EXPECTED_N_ARGS=20
if [ "$NUM_ARGUMENTS" -ne ${EXPECTED_N_ARGS} ]; then
    help
    return
fi

#eval set -- "$OPTS"

while :
do
  case "$1" in
    --dataset_root )
      DATASET_ROOT="$2"
      shift 2
      ;;
    --datasource )
      DATA_SOURCE="$2"
      shift 2
      ;;
    --downloaded_folder )
      DOWNLOADED_FOLDER="$2"
      shift 2
      ;;
    --source_folder )
      SOURCE_FOLDER="$2"
      shift 2
      ;;
    --output_folder )
      OUTPUT_FOLDER="$2"
      shift 2
      ;;
    --split_types )
      SPLIT_TYPES="$2"
      shift 2
      ;;
    --pair_types )
      PAIR_TYPES="$2"
      shift 2
      ;;
    --tokenizer_name )
      TOKENIZER_NAME="$2"
      shift 2
      ;;
    --annotated_data_dir )
      ANNOTATED_DATA_DIR="$2"
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
#source ~/dev/pyvenv/bin/activate
export PYTHONPATH="$PYTHONPATH:$PWD:$PWD/..:$PWD/../.."
export PATH=/usr/local/cuda-11.3/bin:$PATH
export CPATH=/usr/local/cuda-11.3/include:$CPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/lib64
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO


RUN_TRACE_DIR="${DATASET_ROOT}/run_trace"
[ -d ${RUN_TRACE_DIR} ] || mkdir -p ${RUN_TRACE_DIR}


#WORKING_FOLDER_NAME="`basename $PWD`"

WHICH_TYPE=$( echo "${SPLIT_TYPES}" | cut -d '[' -f 2 | cut -d ']' -f 1 )
WHICH_TYPE=$( echo "${WHICH_TYPE}" | cut -d '"' -f 2 | cut -d '"' -f 1 )

today=`date '+%Y_%m_%d_%H_%M'`;
LOGFILE="${RUN_TRACE_DIR}/${DATA_SOURCE}_build_${WHICH_TYPE}_doc_only_$today.log"
echo ${LOGFILE}
echo $HOSTNAME >${LOGFILE}

DOWNLOADED_DIR="${DATASET_ROOT}/${DATA_SOURCE}/${DOWNLOADED_FOLDER}"
SOURCE_DIR="${DATASET_ROOT}/${DATA_SOURCE}/${ANNOTATED_DATA_DIR}/${SOURCE_FOLDER}"
OUTPUT_DIR="${DATASET_ROOT}/${DATA_SOURCE}/${ANNOTATED_DATA_DIR}/${DATA_SOURCE}/${OUTPUT_FOLDER}"
[ -d ${OUTPUT_DIR} ] || mkdir -p ${OUTPUT_DIR}

echo "dataset_root:         ${DATASET_ROOT}"
echo "downloaded_dir:       ${DOWNLOADED_DIR}"
echo "source_dir:           ${SOURCE_DIR}"
echo "output_dir:           ${OUTPUT_DIR}"
echo "split_types:          ${SPLIT_TYPES}"
echo "pair_types:           ${PAIR_TYPES}"
echo "tokenizer_name:       ${TOKENIZER_NAME}"
echo "annotated_data_dir:   ${ANNOTATED_DATA_DIR}"


build_args="--downloaded_dir ${DOWNLOADED_DIR} --src_dir ${SOURCE_DIR} --output_dir ${OUTPUT_DIR} --split_types ${SPLIT_TYPES} --pair_types ${PAIR_TYPES} --tokenizer_name ${TOKENIZER_NAME}"

nohup python3 -u annotated_doc_dataset_builder.py ${build_args} >>${LOGFILE} 2>&1 &
