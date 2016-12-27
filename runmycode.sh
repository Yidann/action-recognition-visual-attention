rm -rf ucf11info
cp -r /shared/tmp/wyd/ucf11info .
THEANO_FLAGS='floatX=float32,device=gpu0,mode=FAST_RUN,nvcc.fastmath=True' python -m scripts.evaluate_ucf11
