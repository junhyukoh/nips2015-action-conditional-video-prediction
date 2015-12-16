if [ -z "${4}" ] 
  then echo "[weights (.caffemodel)] [num_action] [num_step] [gpu]"; exit 0
fi

MODEL=2
WEIGHTS=${1}
ACT=${2}
K=11
STEP=${3}
GPU=${4}

PYTHONPATH=$PWD/../caffe/python
python ../test.py --model $MODEL --weights $WEIGHTS --num_act ${ACT} --K $K --num_step $STEP --gpu $GPU ${5} ${6} ${7} ${8} ${9} ${10} ${11} ${12} ${13} ${14} ${15} ${16} ${17} ${18} ${19} ${20}
