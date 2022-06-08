from __future__ import absolute_import, division, print_function
import os.path as op
import numpy as np
import pandas as pd
import numpy.testing as npt
import jax.numpy as jnp
import haiku_template as sb

data_path = op.join(sb.__path__[0], 'data')

def test_haiku_model_wrapper():
    cfg = {'linear':16}
    class SimpleLinear(hk.Module):
        def __init__(self, cfg, name=None):
            super().__init__(name=name)
            self.cfg = cfg
        def __call__(self, x, analysis, debug):
            return hk.Linear(self.cfg['linear'])(x)

    model = sb.modules.HaikuAutoInit(cfg, SimpleLinear)
    print(model(jnp.random.randn(28, 3)))


# def test_transform_data():
#     """
#     Testing the transformation of the data from raw data to functions
#     used for fitting a function.

#     """
#     # We start with actual data. We test here just that reading the data in
#     # different ways ultimately generates the same arrays.
#     ortho = pd.read_csv(op.join(data_path, 'ortho.csv'))
#     x1, y1, n1 = sb.transform_data(ortho)
#     x2, y2, n2 = sb.transform_data(op.join(data_path, 'ortho.csv'))
#     npt.assert_equal(x1, x2)
#     npt.assert_equal(y1, y2)
#     # We can also be a bit more critical, by testing with data that we
#     # generate, and should produce a particular answer:
#     my_data = pd.DataFrame(
#         np.array([[0.1, 2], [0.1, 1], [0.2, 2], [0.2, 2], [0.3, 1],
#                   [0.3, 1]]),
#         columns=['contrast1', 'answer'])
#     my_x, my_y, my_n = sb.transform_data(my_data)
#     npt.assert_equal(my_x, np.array([0.1, 0.2, 0.3]))
#     npt.assert_equal(my_y, np.array([0.5, 0, 1.0]))
#     npt.assert_equal(my_n, np.array([2, 2, 2]))
