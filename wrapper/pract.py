import pandas as pd
import numpy as np
from wrapper import common

class Pract:

    def DDM(self, d, r, g):
        p = d / (r - g)
        return(p)

    def DCF(self, r, *cf):   ### 구글링해보니 *가 불특정 n개를 의미
        n = 1   #당연히 기말기준
        p = 0  ## 초기값 주고
        for c in cf:
            p = p + (c / (1+r)**n)
            n = n + 1
        return(p)

