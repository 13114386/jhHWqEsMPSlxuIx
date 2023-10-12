## Python virtual environment
    Refer to pyvenv_requirements.txt.


## Process to build datasets
- Download CNNDM and XSum datasets into separate folders such as cnndm/ and xsum/.<br>
  (You may use Huggingface datasets APIs to get them).
- CNNDM contain:
    - train.json (287113 lines)
    - validation.json (13368 lines)
    - test.json (11490 lines)
- XSum contain:
    - train.json (204045 lines)
    - validation.json (11332 lines)
    - test.json (11334 lines)


### Note: all following running script examples should be one-line commands without backslash. Multiple lined examples here are for readability.

### For train dataset
#### Annotate syntactic structures
  Note: We use Stanford CoreNlp4.4.0 for that.
1. Download Stanford CoreNlp4.4.0 and unzip it.
2. Run CoreNlp4.4.0 server in the unzipped folder as follows,<br>
     *nohup java -mx8g -cp "\*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port $1 -timeout 60000 2>corenlp_svr_log_$1.out >/dev/null &*
   - Note: replace $1 with an available port number (e.g., 9000).
3. cd data_preprocess/base
4. Run corenlp_annotate.sh<br>
   An example for CNNDM:
      ```
      . corenlp_annotate.sh \
            --port              9000 \ # same port number as the running Stanford CoreNlp server.
            --datasource        cnndm \
            --dataset_root      /root/dataset/set/to/be/common/to/datasets/directories/or/folders \
            --downloaded_folder folder/to/downloaded/cnndm/dataset \
            --split_type        "train" \
            --column_names      "[article,highlights]" \
            --mode              PROP_NER_COREF \
            --output_dir        a_parent_folder/to/output_folder \
            --python_venv_dir   /your/python/venv/bin/directory
      ```
   An example for XSum:
      ```
      . corenlp_annotate.sh \
            --port              9000 \
            --datasource        xsum \
            --dataset_root      /root/dataset/set/to/be/common/to/datasets/directories/or/folders \
            --downloaded_folder folder/to/downloaded/xsum/dataset \
            --split_type        "train" \
            --column_names      "[document,summary]" \
            --mode              PROP_NER_COREF \
            --output_dir        a_parent_folder/to/output_folder \
            --python_venv_dir   /your/python/venv/bin/directory
      ```
#### Build runtime train datasets from the annotated datasets.
- An example for CNNDM:
    ```
    . annotated_dataset_builder.sh \
        --dataset_root /root/dataset/set/to/be/common/to/datasets/directories/or/folders \
        --datasource cnndm \
        --downloaded_folder directory/to/downloaded/cnndm \
        --source_folder directory/to/stanford_corenlp_annotated_output_files \
        --output_folder folder/to/output_files \
        --split_types "[train]" \
        --pair_types "[article,highlights]" \
        --tokenizer_name facebook/bart-base \
        --build_vocab true \
        --build_compose true \
        --reconcile_vocab false \
        --build_stype2id true \
        --use_source_corefs true \
        --use_target_corefs true \
        --annotated_data_dir an_optinal_parent_folder/to/above_output_folder_or_set_to_dot
    ```
- An example for XSum:<br>
  Note: Our work requires two-phased fine-tuning for XSum. So, its build process has multiple iterations.
    - Build dataset (large sample set) without coreference for first phase fine-tuning:
        ```
        . annotated_dataset_builder.sh \
            --dataset_root /root/dataset/common/to/all/rest/of/directories/or/folders \
            --datasource xsum \
            --downloaded_folder directory/to/downloaded/xsum \
            --source_folder directory/to/stanford_corenlp_output_files \
            --output_folder folder1/to/output_files \
            --split_types "[train]" \
            --pair_types "[document,summary]" \
            --tokenizer_name facebook/bart-base \
            --build_vocab true \
            --build_compose true \
            --reconcile_vocab false \
            --build_stype2id false \
            --use_source_corefs false \
            --use_target_corefs false \
            --annotated_data_dir a_parent_folder/to/folder1/to/output_files
        ```
    - Build dataset (smaller sample set) with coreference for second phase fine-tuning:
        ```
        . annotated_dataset_builder.sh \
            --dataset_root /root/dataset/common/to/all/rest/of/directories/or/folders \
            --datasource xsum \
            --downloaded_folder directory/to/downloaded/xsum \
            --source_folder directory/to/stanford_corenlp_output_files \
            --output_folder folder2/to/output_files \
            --split_types "[train]" \
            --pair_types "[document,summary]" \
            --tokenizer_name facebook/bart-base \
            --build_vocab true \
            --build_compose true \
            --reconcile_vocab false \
            --build_stype2id false \
            --use_source_corefs true \
            --use_target_corefs true \
            --annotated_data_dir a_parent_folder/to/folder2/to/output_files
        ```
    - Reconcile struct feature vocabs from the two built datasets above.<br>
      Note: the process recursively searches from the annotated_data_dir to find the related structure feature vocabs and merge them into a consolidated one.
        ```
        . annotated_dataset_builder.sh \
            --dataset_root /root/dataset/common/to/all/rest/of/directories/or/folders \
            --datasource xsum \
            --downloaded_folder "." \
            --source_folder "." \
            --output_folder "." \
            --split_types "[train]" \
            --pair_types "[document,summary]" \
            --tokenizer_name facebook/bart-base \
            --build_vocab false \
            --build_compose false \
            --reconcile_vocab true \
            --build_stype2id false \
            --use_source_corefs false \
            --use_target_corefs false \
            --annotated_data_dir a_parent_folder/to/above_two_output_folders
        ```
    - Final step: Convert structure feature labels into ids using the reconciled vocabularies.<br>
      Note: This is done for each above dataset separately.
      An example:
        ```
        . annotated_dataset_builder.sh \
            --dataset_root /root/dataset/common/to/all/rest/of/directories/or/folders \
            --datasource xsum \
            --downloaded_folder directory/to/downloaded/xsum \
            --source_folder directory/to/stanford_corenlp_output_files \
            --output_folder folder1/to/output_files \
            --split_types "[train]" \
            --pair_types "[document,summary]" \
            --tokenizer_name Facebook/bart-base \
            --build_vocab false \
            --build_compose false \
            --reconcile_vocab false \
            --build_stype2id true \
            --use_source_corefs false \
            --use_target_corefs false \
            --annotated_data_dir a_parent_folder/to/folder1/to/output_files
        ```

#### Pickle datasets
Note: Datasets are quite large. It is time-consuming to load them from JSON files every training run. So, pickle them.
- An example (argument order is important):
    ```
    cd data_preprocess/pickle_dataset
    . pickle_dataset.sh \
        --dataset_root /directory/to/an_above_built_dataset \
        --dataset_folder an_above_built_dataset_folder \
        --split_type "train" \
        --pair_type "document,summary" \
        --ext_type dataset.json \
        --python_venv_dir /your/python/venv/bin/directory
    ```

### For validation and test datasets
#### Annotate dataset
Note: Stanford CoreNLP uses Penn Treebank for parsing structures. To ensure data distribution consistency with the train dataset, we also annotate validation and test. But both validation and test datasets do not require extra structures. So, the annotation process uses a smaller set of features for both CNNDM and XSum.<br>
- An example for building CNNDM validation set:
    ```
    . corenlp_annotate.sh \
        --port              9000 \
        --datasource        cnndm \
        --dataset_root      /root/dataset/set/to/be/common/to/datasets/directories/or/folders \
        --downloaded_folder folder/to/downloaded/cnndm/dataset \
        --split_type        "validation" \
        --column_names      "[article,highlights]" \
        --mode              PROP_DEFAULT \
        --output_dir        a_parent_folder/to/output_folder \
        --python_venv_dir   /your/python/venv/bin/directory
    ```
Note: 
1. same port number as the running Stanford CoreNlp server.
2. to annotate test set, please replace "validation" with "test".
3. Annotate XSum is similar.

#### Build runtime dataset from annotated datasets.
- An example for CNNDM validation set:
    ```
    . annotated_doc_dataset_builder.sh \
        --dataset_root /root/dataset/set/to/be/common/to/datasets/directories/or/folders \
        --datasource cnndm \
        --downloaded_folder directory/to/downloaded/cnndm \
        --source_folder directory/to/stanford_corenlp_annotated_output_files \
        --output_folder folder/to/output_files" \
        --split_types "[validation]" \
        --pair_types "[article,highlights]" \
        --tokenizer_name facebook/bart-base \
        --annotated_data_dir an_optinal_parent_folder/to/above_output_folder_or_set_to_dot \
        --python_venv_dir    /your/python/venv/bin/directory
    ```
Note: The process is applicable for XSum.

## Source code:
    src/ contains source code.
    src/ats_mllts/run.phase.mgpu.sh is the bash shell runtime script.
Note: we use Huggingface's Accelerate package for multi-GPU parallelism.

### Runtime example for CNNDM
- finetune:
    ```
    . run.phase.mgpu.sh \
        --run_session_type train \
        --datasource cnn \
        --alias cnndm \
        --dataset_root  /root/directory/to/runtime/dataset/folders \
        --token_type "" \
        --data_build_type struct.ner_coref\
        --pretrained_model facebook/bart-base \
        --split_type "[\"train\",\"validation\"]" \
        --pair_type "[\"article\",\"highlights\"]" \
        --data_file_type struct \
        --sd_vocab_folder runtime/dataset/folder \
        --coref_vocab_folder runtime/dataset/folder \
        --struct_feature_level coref \
        --dataset_changed false \
        --query_model_size true \
        --python_venv_dir /your/python/venv/bin/directory
    ```
- test:
    ```
    . run.phase.mgpu.sh \
        --run_session_type test \
        --datasource cnn \
        --alias cnndm \
        --dataset_root  /root/directory/to/runtime/dataset/folders \
        --token_type "" \
        --data_build_type struct.ner_coref \
        --pretrained_model facebook/bart-base \
        --split_type "[\"test\"]" \
        --pair_type "[\"article\",\"highlights\"]" \
        --data_file_type struct \
        --sd_vocab_folder runtime/dataset/folder \
        --coref_vocab_folder runtime/dataset/folder \
        --struct_feature_level none \
        --dataset_changed false \
        --query_model_size false \
        --python_venv_dir /your/python/venv/bin/directory
    ```
### Runtime example for XSum
- 1st phase finetune:
    ```
    . run.phase.mgpu.sh \
        --run_session_type train \
        --datasource xsum \
        --alias xsum \
        --dataset_root  /root/directory/to/runtime/dataset/folders \
        --token_type "" \
        --data_build_type struct \
        --pretrained_model facebook/bart-base \
        --split_type "[\"train\",\"validation\"]" \
        --pair_type "[\"document\",\"summary\"]" \
        --data_file_type struct \
        --sd_vocab_folder struct \
        --coref_vocab_folder struct.ner_coref \
        --struct_feature_level struct \
        --dataset_changed false \
        --query_model_size true \
        --python_venv_dir /your/python/venv/bin/directory
    ```
- 2nd phase finetune:
    ```
    . run.phase.mgpu.sh \
        --run_session_type train \
        --datasource xsum \
        --alias xsum \
        --dataset_root  /root/directory/to/runtime/dataset/folders \
        --token_type "" \
        --data_build_type struct.ner_coref \
        --pretrained_model facebook/bart-base \
        --split_type "[\"train\",\"validation\"]" \
        --pair_type "[\"document\",\"summary\"]" \
        --data_file_type struct \
        --sd_vocab_folder struct.ner_coref \
        --coref_vocab_folder struct.ner_coref \
        --struct_feature_level coref \
        --dataset_changed true \
        --query_model_size true \
        --python_venv_dir /your/python/venv/bin/directory
    ```
- test:
    ```
    . run.phase.mgpu.sh \
        --run_session_type test \
        --datasource cnn \
        --alias xsum \
        --dataset_root  /root/directory/to/runtime/dataset/folders \
        --token_type '' \
        --data_build_type struct \
        --pretrained_model facebook/bart-base \
        --split_type "[\"test\"]" \
        --pair_type "[\"document\",\"summary\"]" \
        --data_file_type struct \
        --sd_vocab_folder struct \
        --coref_vocab_folder struct.ner_coref \
        --struct_feature_level none \
        --dataset_changed false \
        --query_model_size false \
        --python_venv_dir /your/python/venv/bin/directory
    ```
