if [ -z "${6}" ] 
  then echo "[model_type (1:CNN, 2:LSTM)] [weights (.caffemodel)] [num_action] [num_input_frames] [num_step] [gpu]"; exit 0
fi

MODEL=${1}
WEIGHTS=${2}
ACT=${3}
K=${4}
STEP=${5}
GPU=${6}

PYTHONPATH=$PWD/../caffe/python
python ../test.py --model $MODEL --weights $WEIGHTS --num_act ${ACT} --K $K --num_step $STEP --gpu $GPU ${7} ${8} ${9} ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20}
