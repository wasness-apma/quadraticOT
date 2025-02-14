from typing import Optional
import numpy as np

def require(test: bool, throwMessage: Optional[str] = None):
    if not test:
        if throwMessage is not None:
            raise Exception(throwMessage)
        else:
            raise Exception("Requirement Failed.")
        
def requireApproxEq(a: float, b: float, epsilon = 10e-5):
    require(np.abs(a - b) < epsilon)


