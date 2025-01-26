from typing import NamedTuple, Callable
from time import time
from multiprocessing import Pool
import logging

import numpy as np

LOGGER = logging.getLogger('main')
LOGGER.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Устанавливаем уровень логирования для обработчика

# Создаем форматтер (необязательно, но полезно)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Добавляем обработчик к логгеру
# LOGGER.addHandler(console_handler)

class CubicEquation(NamedTuple):
    a: np.float64
    b: np.float64
    c: np.float64
    d: np.float64

    def get_value(self, x: np.float64) -> np.float64:
        return self.a * x ** 3 + self.b * x ** 2 + self.c * x + self.d
    
    def get_derivative(self, x: np.float64) -> np.float64:
        return 3 * self.a * x ** 2 + 2 * self.b * x + self.c


def newton(
    f: Callable[[np.float64], np.float64],
    df: Callable[[np.float64], np.float64],
    x0: np.float64,
    max_iter: int,
    tol: float
) -> np.float64:
    i = 0
    while i < max_iter:
        x1 = x0 - f(x0) / df(x0)
        if np.isclose(x1, x0, atol=tol):
            return x1
        x0 = x1
        i += 1
    return x1


def root_n(
    x: np.complex128,
    n: int
)-> set[np.complex128]:
    roots = set()
    arg = np.arctan2(x.imag, x.real)
    module = np.abs(x)
    for k in range(n):
        roots.add(
            np.power(module, 1 / n) * np.complex128(
                np.cos((arg + 2 * np.pi * k) / n),
                np.sin((arg + 2 * np.pi * k) / n)
            )
        )
    return roots


def get_roots_by_cardano(
    equation: CubicEquation
)-> list[np.float64]:
    roots: set[np.complex128] = set()
    a, b, c, d = equation

    if a != 0:
        p = np.complex128((3 * a * c - np.pow(b, 2)) / (3 * np.pow(a, 2)), 0)
        q = np.complex128((2 * b ** 3 - 9 * a * b * c + 27 * np.pow(a, 2) * d) / (27 * np.pow(a, 3)), 0)

        Q = np.pow(q / 2, 2) + np.pow(p / 3, 3)
        Qrt = list(root_n(Q, 2))

        alpha = root_n(-q / 2 + Qrt[0], 3)
        beta = root_n(-q / 2 + Qrt[1], 3)

        for i in alpha:
            for j in beta:
                if np.isclose((i * j) + p / 3, 0, atol=1e-18):
                    x = i + j - b / (3 * a)
                    roots.add(x)

    return roots

def test(total):
    cnt_crd = 0
    cnt_npr = 0
    npr_time = 0
    crd_time = 0
    for _ in range(total):
        eq = CubicEquation(
            np.random.uniform(1., 1000.),
            np.random.uniform(-1000., 1000.),
            np.random.uniform(-1000., 1000.),
            np.random.uniform(-1000., 1000.)
        )

        start = time()
        roots_cardano = get_roots_by_cardano(eq)
        crd_time += time() - start

        start = time()
        roots = np.roots(eq)
        npr_time += time() - start

        roots_cardano = [root.real for root in roots_cardano if np.isclose(root.imag, 0, atol=1e-12)]
        roots=[root.real for root in roots if np.isclose(root.imag, 0, atol=1e-12)]

        if roots and all([np.isclose(eq.get_value(root), 0, atol=1e-6)  for root in roots_cardano]):
            cnt_crd += 1
        if roots and all([np.isclose(eq.get_value(root), 0, atol=1e-6)  for root in roots]):
            cnt_npr += 1
    print(f"Cardano time = {crd_time}, \t numpy roots time = {npr_time}")
    return round(cnt_crd / total, 10), round(cnt_npr / total, 10), round((cnt_crd - cnt_npr) / total, 10)

if __name__ == "__main__":
    total = 10 ** 6
    num_procs = 10
    print(test(total))
    # with Pool(num_procs) as p:
        # print(np.sum(p.map(test, [total // num_procs for _ in range(num_procs)])) / num_procs)
