#!/usr/bin/env python

from force_optimizer import optimization
import numpy as np
import time

def rand(min, max):
    """Generate a random float between min and max."""
    return np.random.rand() * (max - min) + min

def initial_optimizations(N=10000, OUTPUT_DIR="initial"):
    for i in range(N):
        # FIXED PARAMETERS
        ARCLENGTH_WEIGHT        = 0.01
        LENGTH_TARGET           = rand(4.5, 5.0)
        UUID_init_from          = None  # not starting from prev. optimization

        # RANDOM PARAMETERS
        order                   = int(np.round(rand(7, 16)))
        R1                      = rand(0.3, 0.75)

        CURVATURE_THRESHOLD     = rand(5, 12)
        MSC_THRESHOLD           = rand(4,6)
        CS_THRESHOLD            = rand(0.166, 0.300)
        CC_THRESHOLD            = rand(0.083, 0.120)
        FORCE_THRESHOLD         = rand(0, 5e+04)
    
        LENGTH_WEIGHT           = 10.0 ** rand(-4, -2)
        CURVATURE_WEIGHT        = 10.0 ** rand(-9, -5)
        MSC_WEIGHT              = 10.0 ** rand(-7, -3)
        CS_WEIGHT               = 10.0 ** rand(-1, 4)
        CC_WEIGHT               = 10.0 ** rand(2, 5)
        FORCE_WEIGHT            = 10.0 ** rand(-14, -9)

        # RUNNING THE JOBS
        res, results, coils = optimization(
            OUTPUT_DIR,
            R1,
            order,
            UUID_init_from,
            LENGTH_TARGET, 
            LENGTH_WEIGHT,
            CURVATURE_THRESHOLD, 
            CURVATURE_WEIGHT,
            MSC_THRESHOLD, 
            MSC_WEIGHT,
            CC_THRESHOLD,
            CC_WEIGHT,
            CS_THRESHOLD,
            CS_WEIGHT,
            FORCE_THRESHOLD,
            FORCE_WEIGHT,
            ARCLENGTH_WEIGHT)
        
        print(f"Job {i+1} completed with UUID={results['UUID']}")
        
initial_optimizations()