if [ -z "${9}" ] 
  then echo "[model_type] [result_prefix] [lr] [num_act] [total_time_steps] [num_input_frames] [num_step] [batch_size] [gpu]"; exit 0
fi

MODEL=${1}
PREFIX=${2}
LR=${3}
ACT=${4}
T=${5}
K=${6}
STEP=${7}
BATCH_SIZE=${8}
GPU=${9}

PYTHONPATH=$PWD/../caffe/python
python ../train.py --model $MODEL --prefix $PREFIX --lr $LR --num_act ${ACT} --T $T --K $K --num_step $STEP --batch_size $BATCH_SIZE --gpu $GPU ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20}
