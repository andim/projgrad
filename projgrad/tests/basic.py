import numpy as np
import numpy.testing as npt
import projgrad

def test_basic():

    def objective(x):
        f = np.sum(x**2)
        grad = 2 * x
        return f, grad
    res = projgrad.minimize(objective, [0.1, 0.7, 0.2], reltol=1e-8)
    npt.assert_allclose(res.x, np.ones(3)/3.0)
    
if __name__ == '__main__':
    npt.run_module_suite()
