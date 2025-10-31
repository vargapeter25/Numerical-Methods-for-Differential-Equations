import numpy as np
import numpy.typing as npt
from typing import Any, Union
from dataclasses import dataclass
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


@dataclass
class Result:
    step_size: float
    grid: np.ndarray
    y: np.ndarray

    def get_sample(self, sample_size: int) -> Any:
        assert (len(self.grid) - 1) % (sample_size - 1) == 0
        step = (len(self.grid) - 1) / (sample_size - 1)
        return Result(self.step_size * step,
                          self.grid[0:len(self.grid):step],
                          self.y[0:len(self.y):step])

    def plot(self):
        plt.plot(self.grid, self.y, 'r*')
        plt.show()


class RKMethod:
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        """ Initilizes the RK method.

        Checks for matching array sizes, can assert if fails.
        
        :param B: Butcher tableau
        :param b: `b` values
        :param c: `c` values
        """
        self.A = A
        self.stages = np.shape(self.A)[0]
        self.b = b
        self.c = c

        # Check array sizes
        assert np.shape(self.A) == (self.stages, self.stages)
        assert np.shape(self.b) == (self.stages, )
        assert np.shape(self.c) == (self.stages, )


    def is_explicit(self) -> bool:
        """ Checks if the method is explicit.

        It does not rearranges the Butcher tableau, it only works if `B` is in strictly lower triangular format.

        :return: The method is explicit or not.
        """
        return np.allclose(self.A, np.tril(self.A, -1), rtol=0, atol=0)
    

    def calculate_stages(self, f: Any, dim: int, t: float, h: float, y: np.ndarray, k: Union[None, np.ndarray] = None) -> np.ndarray:
        """ Calculates the stages of the RK method in an iteration.
        
        :param f: The function defining the ODE.
        :param dim: Dimension of the problem.
        :param t: The time.
        :param h: The step size.
        :param y: The result of the previous iteration.
        :param k: The stage values, the default value is `None`.
        """
        if k is None:
            k = np.zeros((dim, self.stages))
        else:
            k = np.reshape(k, (dim, self.stages))
        
        for i in range(self.stages):
            k[:,i] = f(t + h * self.c[i], y + np.dot(k, self.A[i,:]) * h)
        return k.flatten()


    def explicit_solver(self, f: Any, y_0: np.ndarray, t_0: float, t_f: float, N: int) -> Result:
        """ Solves the IVP on a uniform grid. Assumes that the RK method is explicit in a way that `is_explicit()` method is true.
        
        Uses the RK method to solve the IVP (`f`, `y_0`, `t_0`) on the uniform grid defined by `t_0`, `t_f` and `N`.

        :param f: The function defining the ODE.
        :param y_0: The initial value at `t_0`.
        :param t_0: The beginning of the time intervall.
        :param t_f: The end of the time intervall.
        :param N: The number of iterations.
        """

        # ODE dim
        ode_dim = np.shape(y_0)[0]

        # step size
        h = (t_f - t_0) / N
        # uniform grid
        t = np.linspace(t_0, t_f, N + 1)
        # convert function result to an np.array
        f_ = lambda t, y : np.asarray(f(t, y))
        # result array
        y = np.zeros((N + 1, ode_dim))
        y[0] = y_0

        for n in range(0, N):
            # calc y[n + 1]
            k = np.reshape(self.calculate_stages(f_, ode_dim, t[n], h, y[n]), (ode_dim, self.stages))
            y[n+1] = y[n] + np.dot(k, self.b) * h
        
        return Result(h, t, y)
    

    def implicit_solver(self, f: Any, y_0: np.ndarray, t_0: float, t_f: float, N: int) -> Result:
        """ Solves the IVP on a uniform grid. Assumes that the RK method is implicit.
        
        Uses the RK method to solve the IVP (`f`, `y_0`, `t_0`) on the uniform grid defined by `t_0`, `t_f` and `N`.

        :param f: The function defining the ODE.
        :param y_0: The initial value at `t_0`.
        :param t_0: The beginning of the time intervall.
        :param t_f: The end of the time intervall.
        :param N: The number of iterations.
        """

        # ODE dim
        ode_dim = np.shape(y_0)[0]

        # step size
        h = (t_f - t_0) / N
        # uniform grid
        t = np.linspace(t_0, t_f, N + 1)
        # convert function result to an np.array
        f_ = lambda t, y : np.asarray(f(t, y))
        # result array
        y = np.zeros((N + 1, ode_dim))
        y[0] = y_0

        for n in range(0, N):
            # calc y[n + 1]
            # function for non linear solver
            f_helper = lambda k : self.calculate_stages(f_, ode_dim, t[n], h, y[n], k) - k
            k = fsolve(f_helper, np.zeros((ode_dim, self.stages)))
            y[n+1] = y[n] + np.dot(k, self.b) * h
        
        return Result(h, t, y)


    def solver(self, f: Any, y_0: np.ndarray, t_0: float, t_f: float, N: int) -> Result:
        """ Solves the IVP on a uniform grid.
        
        Uses the RK method to solve the IVP (`f`, `y_0`, `t_0`) on the uniform grid defined by `t_0`, `t_f` and `N`.

        :param f: The function defining the ODE.
        :param y_0: The initial value at `t_0`.
        :param t_0: The beginning of the time intervall.
        :param t_f: The end of the time intervall.
        :param N: The number of iterations.
        """

        if self.is_explicit():
            return self.explicit_solver(f, y_0, t_0, t_f, N)
        else:
            return self.implicit_solver(f, y_0, t_0, t_f, N)

    # TODO: A-stability region + function (+ is A-stable), possible maximal order, order


def get_exact_result(f: Any, dim: int, t_0: float, t_f: float, N: int) -> Result:
    h = (t_f - t_0) / N
    # uniform grid
    t = np.linspace(t_0, t_f, N + 1)
    # convert function result to an np.array
    f_ = lambda t: np.asarray(f(t))
    # result array
    y = np.zeros((N + 1, dim))

    for n in range(N + 1):
        y[n] = f_(t[n])
    
    return Result(h, t, y)
