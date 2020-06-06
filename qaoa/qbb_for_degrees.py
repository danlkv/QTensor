import sys
sys.path.append('..')
import qtree as qt
import utils_qaoa as qu
from multiprocessing.dummy import Pool
import numpy as np


def main():
    pool = Pool(processes=100)

    sizes = np.arange(10,15)




if __name__=='__main__':
    main()
