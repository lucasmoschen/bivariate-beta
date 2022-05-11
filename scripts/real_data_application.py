#usr/bin/env python3
"""
Real data application 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from __init__ import ROOT_DIR
import os

def main():

    data = pd.read_csv(os.path.join(ROOT_DIR, '../data/forest_pollen_counts.csv'))
    

if __name__ == '__main__':

    main()
