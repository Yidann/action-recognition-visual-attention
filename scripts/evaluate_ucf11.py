import numpy
import sys
import argparse

#import util.gpu_util
#board = util.gpu_util.LockGPU()
#print 'GPU Lock Acquired'

from src.actrec import train

def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params

    trainerr, validerr, testerr = train(dim_out=params['dim_out'][0],
                                        ctx_dim=params['ctx_dim'][0],
                                        dim=params['dim'][0],
                                        n_actions=params['n_actions'][0],
                                        n_layers_att=params['n_layers_att'][0],
                                        n_layers_out=params['n_layers_out'][0],
                                        n_layers_init=params['n_layers_init'][0],
                                        ctx2out=params['ctx2out'][0],
                                        patience=params['patience'][0],
                                        max_epochs=params['max_epochs'][0],
                                        dispFreq=params['dispFreq'][0],
                                        decay_c=params['decay_c'][0],
                                        alpha_c=params['alpha_c'][0],
                                        temperature_inverse=params['temperature_inverse'][0],
                                        lrate=params['learning_rate'][0],
                                        selector=params['selector'][0],
                                        maxlen=params['maxlen'][0],
                                        optimizer=params['optimizer'][0], 
                                        batch_size=params['batch_size'][0],
                                        valid_batch_size=params['valid_batch_size'][0],
                                        saveto=params['model'][0],
                                        validFreq=params['validFreq'][0],
                                        saveFreq=params['saveFreq'][0],
                                        dataset=params['dataset'][0], 
                                        dictionary=params['dictionary'][0],
                                        use_dropout=params['use_dropout'][0],
                                        reload_=params['reload'][0],
					training_stride=params['training_stride'][0],
					testing_stride=params['testing_stride'][0],
                                        last_n=params['last_n'][0],
                                        fps=params['fps'][0]
                             )
    return validerr

if __name__ == '__main__':
    options = {
        'dim_out': [24],		# hidden layer dim for outputs
        'ctx_dim': [512],		# context vector dimensionality
        'dim': [24],			# the number of LSTM units
        'n_actions': [11],		# number of digits to predict
        'n_layers_att':[1],
        'n_layers_out': [1],
        'n_layers_init': [1],
        'ctx2out': [False],
        'patience': [10],
        'max_epochs': [10],
        'dispFreq': [20],
        'decay_c': [0.0], 
        'alpha_c': [0.0], 
        'temperature_inverse': [1],
        'learning_rate': [0.0001],
        'selector': [True],
        'maxlen': [15],
        'optimizer': ['adam'],
        'batch_size': [128],
        'valid_batch_size': [128],
        'model': ['model_ucf11.npz'],
        'validFreq': [100],
        'saveFreq': [100],		# save the parameters after every saveFreq updates
        'dataset': ['ucf11'],
        'dictionary': [None],
        'use_dropout': [True],
        'reload': [False],
	'training_stride': [1],
	'testing_stride': [1],
        'last_n': [4],			# timesteps from the end used for computing prediction
        'fps': [10]
    }

    if len(sys.argv) > 1:
        options.update(eval("{%s}"%sys.argv[1]))

    main(0, options)
    #util.gpu_util.FreeGPU(board)
    #print 'GPU freed'

