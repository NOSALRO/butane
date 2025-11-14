from typing import Union, Optional, Callable, Tuple
import torch
import scipy

class OTPlanner:

    def __init__(
        self,
        cost_func: Union[str, Callable] = 'l2_squared',
        condition_cost_func: Optional[Union[str, Callable]] = None,
    ):
        _cost_func_map = dict(
            l2_squared = lambda x1, x2: torch.cdist(x1, x2).pow(2),
            l2 = lambda x1, x2: torch.cdist(x1, x2)
        )
        if isinstance(cost_func, str):
            self.cost_func = _cost_func_map[cost_func]
        elif callable(cost_func):
            self.cost_func = cost_func

        if condition_cost_func is not None:
            _condition_cost_func_map = dict(
                l2_squared = lambda c1, c2: torch.cdist(c1, c2).pow(2),
                cosine = lambda c1, c2: 1 - torch.mm(
                    torch.nn.functional.normalize(c1, dim=-1),
                    torch.nn.functional.normalize(c2, dim=-1).transpose(0, 1),
                )
            )
            if isinstance(condition_cost_func, str):
                self.condition_cost_func = _condition_cost_func_map[condition_cost_func]
            elif callable(condition_cost_func):
                self.condition_cost_func = condition_cost_func

    def exact_ot(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        unoptimal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x1.size() == x2.size(), "X1 and X2 should be the same size!"
        c_matrix = self.cost_func(x1.view(x1.size(0), -1).float(), x2.view(x2.size(0), -1).float()).cpu()
        ot_map_i, ot_map_j = scipy.optimize.linear_sum_assignment(c_matrix if not unoptimal else -c_matrix)
        return x1[ot_map_i], x2[ot_map_j]

    def c2_ot(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        r: Optional[float] = None,
        w: Optional[float] = None,
        max_iters: int = 10,
        max_w: float = 1e+09,
        tol: float = 1e-08,
        unoptimal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        def _calc_r(Cx, Cc, diag, w: float):
            c_matrix_prime = Cx + w * Cc
            return c_matrix_prime, (c_matrix_prime <= diag).float().mean()

        def _search_w(Cx, Cc):

            lower = 0.
            upper = w
            diag = torch.diag(Cx)

            c_matrix, r_new = _calc_r(Cx, Cc, diag, 0)
            if r_new < r:
                return c_matrix, 0

            for _ in range(max_iters):
                _, r_new = _calc_r(Cx, Cc, diag, upper)
                if r_new > r:
                    lower = upper
                    upper *= 2
                else:
                    break
                if upper > max_w :
                    return (Cx + max_w * Cc), max_w

            for _ in range(max_iters):
                m = (lower + upper) / 2
                c_matrix, r_new = _calc_r(Cx, Cc, diag, m)
                if r_new < r:
                    upper = m
                else:
                    lower = m
                if abs(r_new - r) < tol:
                    return c_matrix, m
            return c_matrix, m

        assert x1.size() == x2.size(), "X1 and X2 should be the same size!"
        x_c_matrix = self.cost_func(x1.view(x1.size(0), -1).float(), x2.view(x2.size(0), -1).float()).cpu()

        assert c1.size() == c2.size(), "C1 and C2 should be the same size!"
        c_c_matrix = self.condition_cost_func(c1.view(c1.size(0), -1).float(), c2.view(c2.size(0), -1).float()).cpu()

        if r is not None:
            c_matrix, _w = _search_w(x_c_matrix, c_c_matrix)
        else:
            c_matrix = x_c_matrix + w * c_c_matrix


        ot_map_i, ot_map_j = scipy.optimize.linear_sum_assignment(c_matrix if not unoptimal else -c_matrix)
        return x1[ot_map_i], x2[ot_map_j], c1[ot_map_i], c2[ot_map_j]
