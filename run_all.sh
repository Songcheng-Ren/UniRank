#!/usr/bin/env bash
# =========================================================================
# Copyright (C) 2026. UniRank Authors. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

set -euo pipefail

# 进入脚本所在目录，避免从别处启动时相对路径混乱
cd "$(dirname "$0")"

# 使用 4 张 GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG_DIR="./config"
NPROC=4

# 为每次 torchrun 分配不同端口，避免端口残留冲突
BASE_PORT=29500
RUN_IDX=0

# 重试配置
# MAX_RETRIES=-1 表示无限重试；=3 表示最多重试 3 次（总共最多执行 4 次：1次初始 + 3次重试）
MAX_RETRIES=1
RETRY_WAIT_SECONDS=10

# 日志目录
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

# 临时目录根路径：必须尽量短。Python multiprocessing 会在 TMPDIR 下创建
# AF_UNIX socket，路径过长会触发 "OSError: AF_UNIX path too long"。
TEMP_ROOT="$(mktemp -d /tmp/ur.XXXXXX)"

# 总日志文件
TIMESTAMP=$(date '+%F_%H-%M-%S')
MASTER_LOG="${LOG_DIR}/run_all_${TIMESTAMP}.log"

cleanup_temp_path() {
    local path="${1:-}"
    if [[ -n "${path}" && "${path}" == "${TEMP_ROOT}/"* && -d "${path}" ]]; then
        rm -rf "${path}" 2>/dev/null || true
    fi
}

cleanup_temp_dirs() {
    if [[ -d "${TEMP_ROOT}" ]]; then
        find "${TEMP_ROOT}" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
        rmdir "${TEMP_ROOT}" 2>/dev/null || true
    fi
}

prepare_run_temp_dir() {
    local expid="$1"
    local attempt="$2"
    local run_temp_dir="${TEMP_ROOT}/t${RUN_IDX}_a${attempt}"
    mkdir -p "${run_temp_dir}"
    printf '%s\n' "${run_temp_dir}"
}

cleanup() {
    echo "[$(date '+%F %T')] Cleaning temporary files..."
    cleanup_temp_dirs
}
trap cleanup EXIT INT TERM

run_exp() {
    local expid="$1"
    local exp_log="${LOG_DIR}/${expid}_${TIMESTAMP}.log"
    local attempt=0

    while true; do
        attempt=$((attempt + 1))
        local port=$((BASE_PORT + RUN_IDX))
        RUN_IDX=$((RUN_IDX + 1))

        echo "=================================================="
        echo "[$(date '+%F %T')] Starting experiment: ${expid} | attempt=${attempt} | master_port=${port}"
        echo "=================================================="

        # 每次实验前清理一次旧的 torch/torchelastic 临时目录
        cleanup_temp_dirs
        local run_temp_dir
        run_temp_dir="$(prepare_run_temp_dir "${expid}" "${attempt}")"
        echo "[$(date '+%F %T')] Temporary dir: ${run_temp_dir}"

        # 注意：用 if 包裹失败命令，可以避免 set -e 直接退出脚本
        local rc=0
        if TMPDIR="${run_temp_dir}" \
            TMP="${run_temp_dir}" \
            TEMP="${run_temp_dir}" \
            TORCHINDUCTOR_CACHE_DIR="${run_temp_dir}/torchinductor" \
            TRITON_CACHE_DIR="${run_temp_dir}/triton" \
            TORCH_EXTENSIONS_DIR="${run_temp_dir}/torch_extensions" \
            torchrun \
            --standalone \
            --master_port="${port}" \
            --nproc_per_node="${NPROC}" \
            run_expid.py \
            --config "${CONFIG_DIR}" \
            --expid "${expid}" \
            --gpu "${CUDA_VISIBLE_DEVICES}" 2>&1 | tee -a "${exp_log}"; then
            rc=0
        else
            rc=$?
        fi

        # 一个任务尝试运行结束后，立即清理本次任务的临时目录；不要等 run_all.sh 结束
        cleanup_temp_path "${run_temp_dir}"
        cleanup_temp_dirs

        if [[ "${rc}" -eq 0 ]]; then
            echo "[$(date '+%F %T')] Finished experiment: ${expid} (attempt=${attempt})"
            echo
            break
        else
            echo "[$(date '+%F %T')] ERROR: experiment ${expid} failed (attempt=${attempt}, exit_code=${rc})"

            # 达到最大重试次数则退出整个脚本
            if [[ "${MAX_RETRIES}" -ge 0 && "${attempt}" -gt "${MAX_RETRIES}" ]]; then
                echo "[$(date '+%F %T')] ERROR: experiment ${expid} exceeded max retries (${MAX_RETRIES}). Abort."
                exit "${rc}"
            fi

            echo "[$(date '+%F %T')] Retrying ${expid} after ${RETRY_WAIT_SECONDS}s..."
            sleep "${RETRY_WAIT_SECONDS}"
        fi
    done
}

# 整个脚本输出也写入总日志
exec > >(tee -a "${MASTER_LOG}") 2>&1

# 依次运行：按数据集分组；取消注释即可运行对应实验

# ==================================================
# Taobao_Action experiments
# ==================================================
# run_exp "RankMixer_Taobao_Action"
# run_exp "UniMixer_Taobao_Action"
# run_exp "OneTrans_Taobao_Action"
# run_exp "HyFormer_Taobao_Action"
# run_exp "MixFormer_Taobao_Action"
# run_exp "TokenMixer_Taobao_Action"
# run_exp "HiFormer_Taobao_Action"
# run_exp "INFNet_Taobao_Action"
# run_exp "EST_Taobao_Action"
# run_exp "LONGER_Taobao_Action"
# run_exp "Zenith_Taobao_Action"
# run_exp "HeMix_Taobao_Action"
# run_exp "TokenFormer_Taobao_Action"
# run_exp "UltraHSTU_Taobao_Action"
# run_exp "SSR_Taobao_Action"

# ==================================================
# MerRec_Action experiments
# ==================================================
# run_exp "RankMixer_MerRec_Action"
# run_exp "UniMixer_MerRec_Action"
# run_exp "OneTrans_MerRec_Action"
# run_exp "HyFormer_MerRec_Action"
# run_exp "MixFormer_MerRec_Action"
# run_exp "TokenMixer_MerRec_Action"
# run_exp "HiFormer_MerRec_Action"
# run_exp "INFNet_MerRec_Action"
# run_exp "EST_MerRec_Action"
# run_exp "LONGER_MerRec_Action"
# run_exp "Zenith_MerRec_Action"
# run_exp "HeMix_MerRec_Action"
# run_exp "TokenFormer_MerRec_Action"
# run_exp "UltraHSTU_MerRec_Action"
# run_exp "SSR_MerRec_Action"

# ==================================================
# QK_Video_Action experiments
# ==================================================
# run_exp "RankMixer_QK_Video_Action"
# run_exp "TokenFormer_QK_Video_Action"
# run_exp "UniMixer_QK_Video_Action"
# run_exp "UltraHSTU_QK_Video_Action"
# run_exp "OneTrans_QK_Video_Action"
# run_exp "HyFormer_QK_Video_Action"
# run_exp "MixFormer_QK_Video_Action"
# run_exp "TokenMixer_QK_Video_Action"
# run_exp "HiFormer_QK_Video_Action"
# run_exp "INFNet_QK_Video_Action"
# run_exp "EST_QK_Video_Action"
# run_exp "LONGER_QK_Video_Action"
# run_exp "Zenith_QK_Video_Action"
# run_exp "HeMix_QK_Video_Action"
# run_exp "SSR_QK_Video_Action"

# ==================================================
# KuaiRand_Video_Action experiments
# ==================================================
# run_exp "RankMixer_KuaiRand_Video_Action_Small"
# run_exp "UniMixer_KuaiRand_Video_Action"
# run_exp "OneTrans_KuaiRand_Video_Action"
# run_exp "HyFormer_KuaiRand_Video_Action"
# run_exp "MixFormer_KuaiRand_Video_Action"
# run_exp "TokenMixer_KuaiRand_Video_Action"
# run_exp "HiFormer_KuaiRand_Video_Action"
# run_exp "INFNet_KuaiRand_Video_Action"
# run_exp "EST_KuaiRand_Video_Action"
# run_exp "LONGER_KuaiRand_Video_Action"
# run_exp "Zenith_KuaiRand_Video_Action"
# run_exp "HeMix_KuaiRand_Video_Action"
# run_exp "TokenFormer_KuaiRand_Video_Action"
# run_exp "UltraHSTU_KuaiRand_Video_Action"
# run_exp "SSR_KuaiRand_Video_Action"

# ==================================================
# TencentGR_10M_Action experiments
# ==================================================
# run_exp "RankMixer_TencentGR_10M_Action"
# run_exp "UniMixer_TencentGR_10M_Action"
# run_exp "OneTrans_TencentGR_10M_Action"
# run_exp "HyFormer_TencentGR_10M_Action"
# run_exp "MixFormer_TencentGR_10M_Action"
# run_exp "TokenMixer_TencentGR_10M_Action"
# run_exp "HiFormer_TencentGR_10M_Action"
# run_exp "INFNet_TencentGR_10M_Action"
# run_exp "EST_TencentGR_10M_Action"
# run_exp "LONGER_TencentGR_10M_Action"
# run_exp "Zenith_TencentGR_10M_Action"
# run_exp "HeMix_TencentGR_10M_Action"
# run_exp "TokenFormer_TencentGR_10M_ Action"
# run_exp "UltraHSTU_TencentGR_10M_Action"
# run_exp "SSR_TencentGR_10M_Action"

# ==================================================
# Tokenizer ablations for KuaiRand and MerRec
# RankMixer already uses Chunk and OneTrans already uses Auto, so those
# baseline runs are intentionally omitted.
# ==================================================
#run_exp "RankMixer_KuaiRand_Video_Action_Tokenizer_Auto"
#run_exp "RankMixer_KuaiRand_Video_Action_Tokenizer_Field"
#run_exp "RankMixer_KuaiRand_Video_Action_Tokenizer_Random"
#run_exp "OneTrans_KuaiRand_Video_Action_Tokenizer_Chunk"
#run_exp "OneTrans_KuaiRand_Video_Action_Tokenizer_Field"
#run_exp "OneTrans_KuaiRand_Video_Action_Tokenizer_Random"

#run_exp "RankMixer_MerRec_Action_Tokenizer_Auto"
run_exp "RankMixer_MerRec_Action_Tokenizer_Field"
#run_exp "RankMixer_MerRec_Action_Tokenizer_Random"
#run_exp "OneTrans_MerRec_Action_Tokenizer_Chunk"
run_exp "OneTrans_MerRec_Action_Tokenizer_Field"
#run_exp "OneTrans_MerRec_Action_Tokenizer_Random"

run_exp "RankMixer_KuaiRand_Video_Action_Mid"
run_exp "RankMixer_KuaiRand_Video_Action_Large"
run_exp "RankMixer_KuaiRand_Video_Action_Ultra"

echo "All experiments completed successfully."
