import pyfcfc

import numpy as np

if __name__=='__main__':
    pyfcfc.set_conf(label='[D,R]', has_wt=[1], cnvt=[0], 
                    pc=['DD', 'DR', 'RR'], comp_pc=[1, 0, 1])