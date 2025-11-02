import numpy as np
import numpy.typing as npt
from typing import Any, Union
from dataclasses import dataclass
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from nodepy import runge_kutta_method as nrk


@dataclass
class RKResult:
    step_size: float
    grid: np.ndarray
    y: np.ndarray

    @property
    def size(self) -> int:
        return len(self.grid)

    def get_sample(self, sample_size: int) -> Any:
        """ Returns with a sampled smaller grid.

        It is necessary that `sample_size - 1` divides `size - 1`.

        :param sample_size: The new grid size.
        :return: The new grid solution.
        """
        assert (len(self.grid) - 1) % (sample_size - 1) == 0
        step = (len(self.grid) - 1) // (sample_size - 1)
        return RKResult(self.step_size * step,
                          self.grid[0:len(self.grid):step],
                          self.y[0:len(self.y):step])

    def plot(self):
        plt.plot(self.grid, self.y, 'r*')
        plt.show()


class RKMethod:
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, name: str = 'RK method') -> None:
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
        self.name = name

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
        :return: The calculated stages in a flattened array.
        """
        if k is None:
            k = np.zeros((dim, self.stages))
        else:
            k = np.reshape(k, (dim, self.stages))
        
        for i in range(self.stages):
            k[:,i] = f(t + h * self.c[i], y + np.dot(k, self.A[i,:]) * h)
        return k.flatten()


    def explicit_solver(self, f: Any, y_0: np.ndarray, t_0: float, t_f: float, N: int) -> RKResult:
        """ Solves the IVP on a uniform grid. Assumes that the RK method is explicit in a way that `is_explicit()` method is true.
        
        Uses the RK method to solve the IVP (`f`, `y_0`, `t_0`) on the uniform grid defined by `t_0`, `t_f` and `N`.

        :param f: The function defining the ODE. f must have args in (t, u) order.
        :param y_0: The initial value at `t_0`. Must be an `np.array`.
        :param t_0: The beginning of the time intervall.
        :param t_f: The end of the time intervall.
        :param N: The number of iterations.
        :return: The numberical solution.
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
        y[0] = np.asarray(y_0)

        for n in range(0, N):
            # calc y[n + 1]
            k = np.reshape(self.calculate_stages(f_, ode_dim, t[n], h, y[n]), (ode_dim, self.stages))
            y[n+1] = y[n] + np.dot(k, self.b) * h
        
        return RKResult(h, t, y)
    

    def implicit_solver(self, f: Any, y_0: np.ndarray, t_0: float, t_f: float, N: int) -> RKResult:
        """ Solves the IVP on a uniform grid. Assumes that the RK method is implicit.
        
        Uses the RK method to solve the IVP (`f`, `y_0`, `t_0`) on the uniform grid defined by `t_0`, `t_f` and `N`.

        :param f: The function defining the ODE. f must have args in (t, u) order.
        :param y_0: The initial value at `t_0`. Must be an `np.array`.
        :param t_0: The beginning of the time intervall.
        :param t_f: The end of the time intervall.
        :param N: The number of iterations.
        :return: The numberical solution.
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
        y[0] = np.asarray(y_0)

        for n in range(0, N):
            # calc y[n + 1]
            # function for non linear solver
            f_helper = lambda k : self.calculate_stages(f_, ode_dim, t[n], h, y[n], k) - k
            k = fsolve(f_helper, np.zeros((ode_dim, self.stages)))
            k = np.reshape(k, (ode_dim, self.stages))
            y[n+1] = y[n] + np.dot(k, self.b) * h
        
        return RKResult(h, t, y)


    def solver(self, f: Any, y_0: np.ndarray, t_0: float, t_f: float, N: int) -> RKResult:
        """ Solves the IVP on a uniform grid.
        
        Uses the RK method to solve the IVP (`f`, `y_0`, `t_0`) on the uniform grid defined by `t_0`, `t_f` and `N`.

        :param f: The function defining the ODE. f must have args in (t, u) order.
        :param y_0: The initial value at `t_0`. Must be an `np.array`.
        :param t_0: The beginning of the time intervall.
        :param t_f: The end of the time intervall.
        :param N: The number of iterations.
        :return: The numerical solution.
        """

        if self.is_explicit():
            return self.explicit_solver(f, y_0, t_0, t_f, N)
        else:
            return self.implicit_solver(f, y_0, t_0, t_f, N)

    def meta_data(self) -> None:
        """ Prints information about the RK method.
        
        Show the order of consistency, maximal order, A-stability. Uses the `nodepy` package to calculate the order.
        """

        method_type = 'explicit' if self.is_explicit() else 'implicit'

        output =  f'RK method ({self.name}) meta data:\n'
        output += f'A:\n{self.A}\n'
        output += f'b: {self.b}\n'
        output += f'c: {self.c}\n\n'
        output += f'The method is {method_type}.\n\n'
        output += f'Number of stages: {self.stages}\n\n'
        output += f'Maximum possible order of consistency: {self.maximum_possible_order_of_consistency()}\n\n'
        output += f'Order of consistency: {self.order()}\n'

        print(output)
        self.print_stability_function()
        self.plot_stability_region()
        # If used outside of jupyter: plt.show()

    def get_nodepy_rkm(self) -> Any:
        return nrk.ExplicitRungeKuttaMethod(self.A, self.b, name = self.name) if self.is_explicit() else nrk.RungeKuttaMethod(self.A, self.b, name = self.name)

    def plot_stability_region(self) -> None:
        """ Plots stability region of the RK method.
        
        Uses the `nodepy` package to draw the stability region.
        """
        self.get_nodepy_rkm().plot_stability_region()

    def print_stability_function(self) -> None:
        """ Prints the stability function as a quotient of two polinomials.
        
        Uses the `nodepy` package to calculate the function.
        """
        
        p, q = self.get_nodepy_rkm().stability_function()

        print('Stability function in the form P/Q:\n')
        print('P:\n')
        print(p)
        print('Q:\n')
        print(q)
    
    def order(self) -> int:
        """ Returns with the order of consistency.

        It uses `nodepy` package order method.

        :return: Order of consistency.
        """
        return self.get_nodepy_rkm().order()


    def maximum_possible_order_of_consistency(self) -> int:
        """ Returns the maximam possible order of consistency of the method.
        
        In case of explicit methods if the number of stages is less than `12` it is exact otherwise it gives an upper bound.

        :return: The maximums order.
        """

        assert self.stages >= 1

        if self.is_explicit():
            min_num_of_stages = np.array([1, 2, 3, 4, 6, 7, 9, 11])
            if self.stages <= 11:
                return np.searchsorted(min_num_of_stages, self.stages, side='right')
            
            print('Warning: Stage number is too high for exact result.')
            return self.stages - 2
        
        return self.stages * 2
    

    def order_from_exact_solution(self, f: Any, y_0: np.ndarray, t_0: float, t_f: float, steps: list[int], f_sol: Any, dim_sol: int, dims: np.ndarray | None = None, norm_ord : Any = np.inf) -> np.ndarray:
        """ Estimates the order of consistency from the exact solution.

        Calculates the `E(h_i)` from based on `steps` and the exact solution, and calculates the estimate based on adjacent values.
        
        :param f: The function defining the ODE. f must have args in `(t, u)` order.
        :param y_0: The initial value at `t_0`. Must be an `np.array`.
        :param t_0: The beginning of the time intervall.
        :param t_f: The end of the time intervall.
        :param steps: The number of steps per calculations used in the estimates.
        :param f_sol: The exact solution. f must have an argument `t`.
        :param dim_sol: The dimension of the solution vector.
        :param dims: The dimension used for estimate calcualtion. It must have size `dim_sol`.
        :param norm_ord: The norm used for the approximation. It uses `numpy.norm`.
        :return: An array of the estimates where each column corresponds to one of the coordinates.
        """
        if dims is None:
            dims = np.arange(dim_sol)
        else:
            dims = np.array(dims)

        assert len(np.shape(dims)) == 1
        assert dim_sol == np.size(dims)

        errors = []
        hs = []
        for N in steps:
            exact = get_exact_result(f_sol, dim_sol, t_0, t_f, N)
            rk_sol = self.solver(f, y_0, t_0, t_f, N)

            assert np.shape(rk_sol.y[:,dims]) == np.shape(exact.y)

            hs.append(rk_sol.step_size)
            error = np.linalg.norm(rk_sol.y[:,dims] - exact.y, axis=0, ord=norm_ord)
            # Discrete p-norm | if p is not inf, than
            if isinstance(norm_ord, int):
                error *= rk_sol.step_size**(1./norm_ord)
            errors.append(error)

        result = []
        for i in range(len(errors) - 1):
            result.append(np.log(errors[i] / errors[i+1]) / np.log(hs[i] / hs[i+1]))

        return np.array(result)


    def order_from_fine_grid(self, f: Any, y_0: np.ndarray, t_0: float, t_f: float, steps: list[int], fine_step: int, dims: np.ndarray | None = None, norm_ord : Any = np.inf) -> np.ndarray:
        """ Estimates the order of consistency from a fine grid.

        Calculates the `E(h_i)` from based on `steps` and a fine grid defined by `fine_step`, and calculates the estimate based on adjacent values.
        
        :param f: The function defining the ODE. f must have args in `(t, u)` order.
        :param y_0: The initial value at `t_0`. Must be an `np.array`.
        :param t_0: The beginning of the time intervall.
        :param t_f: The end of the time intervall.
        :param steps: The number of steps per calculations used in the estimates.
        :param fine_step: The number of steps in the fine grid. 
        :param dims: The dimension used for estimate calcualtion. It must have size `dim_sol`.
        :param norm_ord: The norm used for the approximation. It uses `numpy.norm`.
        :return: An array of the estimates where each column corresponds to one of the coordinates.
        """
        if dims is None:
            dims = np.arange(len(y_0))
        else:
            dims = np.array(dims)

        fine_grid_sol = self.solver(f, y_0, t_0, t_f, fine_step)

        errors = []
        hs = []
        for N in steps:
            rk_sol = self.solver(f, y_0, t_0, t_f, N)
            exact = fine_grid_sol.get_sample(len(rk_sol.grid))
            error = np.linalg.norm(rk_sol.y[:, dims] - exact.y[:, dims], axis=0, ord=norm_ord)
            hs.append(rk_sol.step_size)
            # Discrete p-norm | if p is not inf, than
            if isinstance(norm_ord, int):
                error *= rk_sol.step_size**(1./norm_ord)
            errors.append(error)

        result = []
        for i in range(len(errors) - 1):
            result.append(np.log(errors[i] / errors[i+1]) / np.log(hs[i] / hs[i+1]))

        return np.array(result)


    def order_from_coarse_grid_1(self, f: Any, y_0: np.ndarray, t_0: float, t_f: float, N: int, dims: np.ndarray | None = None, norm_ord : Any = np.inf) -> np.ndarray:
        """ Estimates the order of consistency from a coarse grid.

        Calculates the `E(h_i)` for `N`, `2N` and `4N`, approximates with the adjacent values.
        
        :param f: The function defining the ODE. f must have args in `(t, u)` order.
        :param y_0: The initial value at `t_0`. Must be an `np.array`.
        :param t_0: The beginning of the time intervall.
        :param t_f: The end of the time intervall.
        :param N: The number of steps in the original method.
        :param dims: The dimension used for estimate calcualtion. It must have size `dim_sol`.
        :param norm_ord: The norm used for the approximation. It uses `numpy.norm`.
        :return: An array of the estimates where each column corresponds to one of the coordinates.
        """
        if dims is None:
            dims = np.arange(len(y_0))
        else:
            dims = np.array(dims)

        rk_sols = []
        for i in range(3):
            rk_sols.append(self.solver(f, y_0, t_0, t_f, N * 2**i))

        errors = []
        for i in range(2):
            exact = rk_sols[i + 1].get_sample(rk_sols[i].size).y[:,dims]
            error = np.linalg.norm(rk_sols[i].y[:,dims] - exact, axis=0, ord=norm_ord)
            # Discrete p-norm | if p is not inf, than
            if isinstance(norm_ord, int):
                error *= rk_sols[i].step_size**(1./norm_ord)
            errors.append(error)

        order = np.log2(errors[0] / errors[1])
        return order


    def order_from_coarse_grid_2(self, f: Any, y_0: np.ndarray, t_0: float, t_f: float, N: int, dims: np.ndarray | None = None, norm_ord : Any = np.inf) -> np.ndarray:
        """ Estimates the order of consistency from a coarse grid.

        Calculates the `E(h_i)` for `N`, `2N` and `4N`, approximates with the finest grid.
        
        :param f: The function defining the ODE. f must have args in `(t, u)` order.
        :param y_0: The initial value at `t_0`. Must be an `np.array`.
        :param t_0: The beginning of the time intervall.
        :param t_f: The end of the time intervall.
        :param N: The number of steps in the original method.
        :param dims: The dimension used for estimate calcualtion. It must have size `dim_sol`.
        :param norm_ord: The norm used for the approximation. It uses `numpy.norm`.
        :return: An array of the estimates where each column corresponds to one of the coordinates.
        """
        if dims is None:
            dims = np.arange(len(y_0))
        else:
            dims = np.array(dims)

        rk_sols = []
        for i in range(3):
            rk_sols.append(self.solver(f, y_0, t_0, t_f, N * 2**i))

        errors = []
        for i in range(2):
            exact = rk_sols[2].get_sample(rk_sols[i].size).y[:,dims]
            error = np.linalg.norm(rk_sols[i].y[:,dims] - exact, axis=0, ord=norm_ord)
            # Discrete p-norm | if p is not inf, than
            if isinstance(norm_ord, int):
                error *= rk_sols[i].step_size**(1./norm_ord)
            errors.append(error)

        order = np.log2(errors[0] / errors[1] - 1)
        return order


def get_exact_result(f: Any, dim: int, t_0: float, t_f: float, N: int) -> RKResult:
    """ Evaluates the `f` function in the grid points.
    
    f must return an np.ndarray even if the problem is 1 dimensional.

    :param f: The function.
    :param dim: The dimension of the function param.
    :param t_0: The beginning of the time intervall.
    :param t_f: The end of the time intervall.
    :param N: The number of iterations.
    """
    h = (t_f - t_0) / N
    # uniform grid
    t = np.linspace(t_0, t_f, N + 1)
    # convert function result to an np.array
    f_ = lambda t: np.asarray(f(t))
    # result array
    y = np.zeros((N + 1, dim))

    for n in range(N + 1):
        y[n] = f_(t[n])
    
    return RKResult(h, t, y)


# Named RK methods
class RKMethods():
    EE          = RKMethod(np.array([[0]]), np.array([1]), np.array([0]), name='Explicit Euler')
    IE          = RKMethod(np.array([[1]]), np.array([1]), np.array([1]), name='Implicit Euler')
    TRAPEZOIDAL = RKMethod(np.array([[0, 0], [1/2, 1/2]]), np.array([1/2, 1/2]), np.array([0, 1]), name='Trapezoidal')
    RK2         = RKMethod(np.array([[0, 0], [0.5, 0]]), np.array([0, 1]), np.array([0, 0.5]), name='RK2')
    RK4         = RKMethod(np.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]]), np.array([1/6, 1/3, 1/3, 1/6]), np.array([0, 0.5, 0.5, 1]), name='RK4')
