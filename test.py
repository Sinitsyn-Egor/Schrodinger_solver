from dataclasses import dataclass

import numpy as np

import main


@dataclass
class Case:
    input: main.Schrodinger_input
    psi0: np.ndarray[np.complex64] | None
    psi: np.ndarray[np.complex64]


def psi1(t):
    return np.exp(-t)


X, Y, Z, T = np.meshgrid(
    np.array([1, 2, 3, 4, 5]),
    np.array([1, 2, 3, 4, 5]),
    np.array([1, 2, 3, 4, 5]),
    np.array([1, 1.5, 2]),
)
TEST_CASES = [
    Case(
        input=main.Schrodinger_input(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([1, 2, 3, 4, 5]),
            z=np.array([1, 2, 3, 4, 5]),
            t=np.array([1, 2, 3]),
            a=0,
            U=0 * np.ones((5, 5, 5, 3)),
            h=(1, 1, 1),
            dt=1,
        ),
        psi0=1
        * np.pad(np.ones((3, 3, 3)), [(1, 1), (1, 1), (1, 1)], constant_values=0),
        psi=1
        * np.pad(
            np.ones((3, 3, 3, 3)), [(1, 1), (1, 1), (1, 1), (0, 0)], constant_values=0
        ),
    ),
    Case(
        input=main.Schrodinger_input(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([1, 2, 3, 4, 5]),
            z=np.array([1, 2, 3, 4, 5]),
            t=np.array([1, 1.5, 2]),
            a=0,
            U=-1 * np.ones((5, 5, 5, 3)),
            h=(1, 1, 1),
            dt=0.5,
        ),
        psi0=1
        / np.e
        * np.pad(np.ones((3, 3, 3)), [(1, 1), (1, 1), (1, 1)], constant_values=0),
        psi=1
        * np.pad(
            psi1(T)[1:-1, 1:-1, 1:-1],
            [(1, 1), (1, 1), (1, 1), (0, 0)],
            constant_values=0,
        ),
    ),
]

###################
# Tests
###################


def test_find_value(t: Case) -> None:
    psi = main.Schrodinger_solver(t.input, t.psi0)
    assert np.allclose(psi, t.psi, rtol=0.5), f"Test_faild{t.psi, psi}"


for test in TEST_CASES[1:]:
    test_find_value(test)
print("All test passed")
