## Download Model

```bash
BASE_PATH=/home/zp489/scratch/working/v1/2008/15_xtreme
MODEL_TYPE=xlm-roberta-large

# Download XLM-R Large
python jiant/proj/main/export_model.py \
    --model_type ${MODEL_TYPE} \
    --output_base_path ${BASE_PATH}/models/${MODEL_TYPE}
```

## Download Data
```bash
python jiant/scripts/download_data/runscript.py \
    download \
    --benchmark XTREME \
    --output_path ${BASE_PATH}/tasks/
python jiant/scripts/download_data/runscript.py \
    download \
    --tasks mnli squad_v1 \
    --output_path ${BASE_PATH}/tasks/
```

## Tokenize and Cache Data

### XNLI (uses MNLI for training)
```bash
for LG in ar bg de el en es fr hi ru sw th tr ur vi zh; do
    TASK=xnli_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 256 \
        --smart_truncate
done
```

### PAWS-X
```bash
TASK=pawsx_en
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --model_type ${MODEL_TYPE} \
    --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val,test \
    --max_seq_length 256 \
    --smart_truncate
for LG in ar de es fr ja ko zh; do
    TASK=pawsx_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 256 \
        --smart_truncate
done
```

### UDPos
```bash
TASK=udpos_en
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --model_type ${MODEL_TYPE} \
    --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val,test \
    --max_seq_length 256 \
    --smart_truncate
for LG in af ar bg de el es et eu fa fi fr he hi hu id it ja ko mr nl pt ru ta te tr ur vi zh; do
    TASK=udpos_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 256 \
        --smart_truncate
done
for LG in kk th tl yo; do
    TASK=udpos_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases test \
        --max_seq_length 256 \
        --smart_truncate
done
```

### PANX
```bash
TASK=panx_en
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --model_type ${MODEL_TYPE} \
    --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val,test \
    --max_seq_length 256 \
    --smart_truncate
for LG in af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh; do
    TASK=panx_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 256 \
        --smart_truncate
done
```

### XQuAD (uses SQuAD for training)
```bash
for LG in ar de el en es hi ru th tr vi zh; do
    TASK=xquad_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val \
        --max_seq_length 384 \
        --smart_truncate
done
```

### MLQA (uses SQuAD for training)
```bash
for LG in ar de en es hi vi zh; do
    TASK=mlqa_${LG}_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 384 \
        --smart_truncate
done
```

### TyDiQA
```bash
TASK=tydiqa_en
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --model_type ${MODEL_TYPE} \
    --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val \
    --max_seq_length 384 \
    --smart_truncate
for LG in ar bn fi id ko ru sw te; do
    TASK=tydiqa_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val \
        --max_seq_length 384 \
        --smart_truncate
done
```

### Bucc2018
```bash
for LG in de fr ru zh; do
    TASK=bucc2018_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val,test \
        --max_seq_length 512 \
        --smart_truncate
done
```

### Tatoeba
```bash
for LG in af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr nl pt ru sw ta te th tl tr ur vi zh; do
    TASK=tatoeba_${LG}
    python jiant/proj/main/tokenize_and_cache.py \
        --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
        --model_type ${MODEL_TYPE} \
        --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
        --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
        --phases val \
        --max_seq_length 512 \
        --smart_truncate
done
```

### MNLI and SQuAD
```bash
TASK=mnli
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --model_type ${MODEL_TYPE} \
    --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val \
    --max_seq_length 256 \
    --smart_truncate
TASK=squad_v1
python jiant/proj/main/tokenize_and_cache.py \
    --task_config_path ${BASE_PATH}/tasks/configs/${TASK}_config.json \
    --model_type ${MODEL_TYPE} \
    --model_tokenizer_path ${BASE_PATH}/models/${MODEL_TYPE}/tokenizer \
    --output_dir ${BASE_PATH}/cache/${MODEL_TYPE}/${TASK} \
    --phases train,val \
    --max_seq_length 384 \
    --smart_truncate
```

## Generate Run configs
```bash
mkdir -p ${BASE_PATH}/runconfigs

# XNLI
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task xnli \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 2 --train_batch_size 4 --gradient_accumulation_steps 8 \
    --output_path ${BASE_PATH}/runconfigs/xnli.json

# PAWS-X
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task pawsx \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 5 --train_batch_size 4 --gradient_accumulation_steps 8 \
    --output_path ${BASE_PATH}/runconfigs/pawsx.json

# UDPOS
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task udpos \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 10 --train_batch_size 4 --gradient_accumulation_steps 8 \
    --output_path ${BASE_PATH}/runconfigs/udpos.json

# PANX
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task panx \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 10 --train_batch_size 4 --gradient_accumulation_steps 8 \
    --output_path ${BASE_PATH}/runconfigs/panx.json

# XQuAD
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task xquad \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 2 --train_batch_size 4 --gradient_accumulation_steps 4 \
    --output_path ${BASE_PATH}/runconfigs/xquad.json

# MLQA
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task mlqa \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 2 --train_batch_size 4 --gradient_accumulation_steps 4 \
    --output_path ${BASE_PATH}/runconfigs/mlqa.json

# TyDiQA
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task tydiqa \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --epochs 2 --train_batch_size 4 --gradient_accumulation_steps 4 \
    --output_path ${BASE_PATH}/runconfigs/tydiqa.json

# Bucc2018
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task bucc2018 \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --output_path ${BASE_PATH}/runconfigs/bucc2018.json

# Tatoeba
python jiant/scripts/postproc/xtreme/xtreme_runconfig_writer.py \
    --xtreme_task tatoeba \
    --task_config_base_path ${BASE_PATH}/tasks/configs \
    --task_cache_base_path ${BASE_PATH}/cache/${MODEL_TYPE} \
    --output_path ${BASE_PATH}/runconfigs/tatoeba.json
```

## Train/Run models
```bash
for TASK in xnli pawsx udpos panx xquad mlqa tydiqa; do
    python jiant/proj/main/runscript.py \
        run_with_continue \
        --ZZsrc ${BASE_PATH}/models/${MODEL_TYPE}/config.json \
        --jiant_task_container_config_path ${BASE_PATH}/runconfigs/${TASK}.json \
        --model_load_mode from_transformers \
        --learning_rate 1e-5 \
        --eval_every_steps 1000 \
        --no_improvements_for_n_evals 30 \
        --do_save \
        --force_overwrite \
        --do_train --do_val \
        --output_dir ${BASE_PATH}/runs/${TASK}
done

for TASK in bucc2018 tatoeba; do
    python jiant/proj/main/runscript.py \
        run_with_continue \
        --ZZsrc ${BASE_PATH}/models/${MODEL_TYPE}/config.json \
        --jiant_task_container_config_path ${BASE_PATH}/runconfigs/${TASK}.json \
        --model_load_mode from_transformers \
        --force_overwrite \
        --do_val \
        --output_dir ${BASE_PATH}/runs/${TASK}
done
```
