rm model_ucf11.npz
rm model_ucf11.npz.pkl
echo 'runing'
THEANO_FLAGS='floatX=float32,device=gpu0,mode=FAST_RUN,nvcc.fastmath=True' python -m scripts.evaluate_ucf11
