REMAT_ARRAY=("minimal" "full")
USE_INT8_TRAINING_ARRAY=("true")
DTYPE_ARRAY=("bfloat16")
PRNG_KEY_ARRAY=(1 2)

for REMAT_POLICY in ${REMAT_ARRAY[@]}; do
    for USE_INT8_TRAINING in ${USE_INT8_TRAINING_ARRAY[@]}; do
        for DTYPE in ${DTYPE_ARRAY[@]}; do
            for PRNG_KEY in ${PRNG_KEY_ARRAY[@]}; do

                echo "Remat: ${REMAT_POLICY}, int8: ${USE_INT8_TRAINING}, dtype: ${DTYPE}, PRNG_KEY: ${PRNG_KEY}"
                RUN_NAME=mattdavidow-20230725-a2_remat_${REMAT_POLICY}_useint8_${USE_INT8_TRAINING}_dtype_${DTYPE}_PRNGKey_${PRNG_KEY}
                echo "Running the above setting with RUN_NAME ${RUN_NAME}"
                python3 multihost_job.py --BUCKET_NAME="mattdavidow-maxtext-br" --RUN_NAME=${RUN_NAME} --TPU_TYPE=v5litepod-256 --NUM_SLICES=1 --VERSION=v2-alpha-tpuv5-lite --COMMAND="bash setup.sh && bash MaxText/configs/8b-experiments-7-25-23.sh ${REMAT_POLICY} ${USE_INT8_TRAINING} ${DTYPE} ${PRNG_KEY} ${RUN_NAME}" --CQR_EXTRA_ARGS="--reserved --metadata priority-group=medium" --ZONE=us-west4-a
            done        
        done
    done
done