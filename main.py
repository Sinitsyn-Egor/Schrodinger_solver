from dataclasses import dataclass

import numpy as np

PLANK_CONST = 6.62607015e-27  # erg·seconds
MASS_ELECTRON = 9.1093837015e-28  # gramm


@dataclass
class Schrodinger_input:
    x: np.ndarray[np.float64]
    y: np.ndarray[np.float64]
    z: np.ndarray[np.float64]
    t: np.ndarray[np.float64]
    U: np.ndarray[complex]  # x, y, z, t
    a: complex
    h: tuple[float, float, float]  # hx, hy, hz
    dt: float
    # shape: tuple[int, int, int, int] # shape x y z t


def A(Ut: np.ndarray, h: tuple[float, float, float], a: float):
    I, J, K = Ut.shape  # noqa: E741
    hx, hy, hz = h
    N = (I - 2) * (J - 2) * (K - 2)
    di = 1
    dj = I - 2
    dk = (I - 2) * (J - 2)
    A = np.zeros((N, N), dtype=np.complex64)
    for i in range(1, I - 1):
        for j in range(1, J - 1):
            for k in range(1, K - 1):
                l = (i - 1) + (j - 1) * dj + (k - 1) * dk  # noqa: E741

                A[l, l] = 2 * a / hx**2 + 2 * a / hy**2 + 2 * a / hz**2 + Ut[i, j, k]

                if i != I - 2:
                    A[l, l + di] = a / hx**2
                if i != 1:
                    A[l, l - di] = a / hx**2

                if j != J - 2:
                    A[l, l + dj] = a / hy**2
                if j != 1:
                    A[l, l - dj] = a / hy**2

                if k != K - 2:
                    A[l, l + dk] = a / hz**2
                if k != 1:
                    A[l, l - dk] = a / hz**2
    return A


def Schrodinger_solver(
    info: Schrodinger_input, psi0_inp: np.ndarray[np.complex64] | None = None
) -> np.ndarray[np.complex64]:
    xn, yn, zn, tn = info.U.shape
    N = (xn - 2) * (yn - 2) * (zn - 2)
    eye = np.eye(N, N)
    if psi0_inp is None:
        Amat0 = A(info.U[:, :, :, 0], info.h, info.a)
        psi0 = np.linalg.eigh(Amat0)[1][0]
    else:
        assert psi0_inp.shape == (xn, yn, zn), " Wrong psi0 shape"
        psi0 = psi0_inp[1:-1, 1:-1, 1:-1].ravel()

    psi_mas: list[np.ndarray] = [psi0]

    for ti in range(len(info.t) - 1):
        psin = (A(info.U[:, :, :, ti], info.h, info.a) * info.dt + eye) @ psi_mas[ti]
        psi_mas.append(psin)

    psi_mas_reshape = [
        np.pad(psin.reshape(xn - 2, yn - 2, zn - 2), 1, constant_values=0)
        for psin in psi_mas
    ]
    psi = np.stack(psi_mas_reshape, axis=-1)
    return psi


if __name__ == "__main__":
    data = {
        "x": [1, 2, 3, 4, 5],
        "y": [1, 2, 3, 4, 5],
        "z": [1, 2, 3, 4, 5],
        "t": [1, 2, 3],
        "a": 1,
        "U": lambda x, y, z, t: x**2,
    }

    info = Schrodinger_input(
        x=[1, 2, 3, 4, 5],
        y=[1, 2, 3, 4, 5],
        z=[1, 2, 3, 4, 5],
        t=[1, 2, 3],
        a=1j * PLANK_CONST / (2 * MASS_ELECTRON),
        U=-1j * data["U"](*np.meshgrid(data["x"], data["y"], data["z"], data["t"])),
        h=(1, 1, 1),
        dt=1,
    )

    print(info.U.shape)
    res = Schrodinger_solver(info)

    print(res)
    print()
    mod_psi_sq = (res * np.conj(res)).real.astype(np.float32)
    print(np.sum(mod_psi_sq, axis=(0, 1, 2)))
