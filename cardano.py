from typing import NamedTuple
from multiprocessing import Pool

import numpy as np


class CubicEquation(NamedTuple):
    a: np.float64
    b: np.float64
    c: np.float64
    d: np.float64

    def get_value(
        self,
        x: np.float64
    )-> np.float64:
        return self.a * np.pow(x, 3) + self.b * np.pow(x, 2) + self.c * x + self.d


def root_n(
    x: np.complex128,
    n: int
)-> set[np.complex128]:
    roots = set()

    # arg = np.arctan(x.imag / x.real)
    arg = np.arctan2(x.real, x.imag)
    module = np.abs(x)

    for k in range(n):
        roots.add(
            np.power(module, 1 / n) * np.complex128(
                np.cos((arg + 2 * np.pi * k) / n),
                np.sin((arg + 2 * np.pi * k) / n)
            )
        )
    result = {np.complex128(root.real, 0) if np.abs(root.imag) < 1e-12 else root for root in roots}
    return result


def get_roots_by_cardano(
    equation: CubicEquation
)-> list[np.float64]:
    roots: set[np.complex128] = set()
    a, b, c, d = equation

    if a != 0:
        p = np.complex128((3 * a * c - b ** 2) / (3 * a ** 2), 0)
        q = np.complex128((2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3), 0)
        Q = (q / 2) ** 2 + (p / 3) ** 3
        Qrt = list(root_n(Q, 2))
        alpha = root_n(-q / 2 + Qrt[0], 3)
        beta = root_n(-q / 2 + Qrt[1], 3)

        for i in alpha:
            for j in beta:
                if np.abs((i * j) + p / 3) <= 1e-12:
                    x = i + j - b / (3 * a)
                    roots.add(x)
    elif b != 0:
        D = c ** 2 - 4 * b * d
        if D < 0:
            return []
        roots.add((-c - np.sqrt(D)) / 2 * b)
        roots.add((-c + np.sqrt(D)) / 2 * b)
    elif c != 0:
        roots.add(-d / c)
    solution = []
    for root in roots:
        if np.abs(root.imag) < 1e-12:
            solution.append(root.real)
    return solution

def test(total):
    cnt = 0
    for _ in range(total):
        eq = CubicEquation(
            np.random.uniform(-100., 100.),
            np.random.uniform(-100., 100.),
            np.random.uniform(-100., 100.),
            np.random.uniform(-100., 100.)
        )
        roots = get_roots_by_cardano(eq)
        if all([np.abs(eq.get_value(root)) < 1e-9  for root in roots]):
            cnt += 1
    return cnt / total

if __name__ == "__main__":
    total = 10 ** 6
    num_procs = 10

    with Pool(num_procs) as p:
        print(np.sum(p.map(test, [total // num_procs for _ in range(num_procs)])) / num_procs)
