from __future__ import annotations

import math
import struct
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


Number = float
Matrix = List[List[Number]]
Vector = List[Number]


__all__ = [
    "to_float32",
    "round_sig",
    "back_substitution",
    "gaussian_elimination",
    "solve_gauss",
    "rref",
    "solve_gauss_jordan",
    "classify_linear_system",
    "alpha_system_matrix",
    "alpha_system_rhs",
    "analyze_alpha_system",
    "matvec",
    "vec_sub",
    "food_slack",
    "food_is_enough",
    "max_individual_increase",
    "extinction_max_increase",
    "check_solution",
    "self_test",
]


def to_float32(x: Number) -> Number:
    """Convert a Python float to IEEE-754 single precision and back."""
    return struct.unpack("!f", struct.pack("!f", float(x)))[0]


def round_sig(x: Number, sig: int) -> Number:
    """Round a number to `sig` significant digits."""
    if sig <= 0:
        raise ValueError("sig must be >= 1")
    x = float(x)
    if x == 0.0:
        return 0.0
    sign = -1.0 if x < 0 else 1.0
    x = abs(x)
    exp = math.floor(math.log10(x))
    factor = 10 ** (sig - 1 - exp)
    return sign * (round(x * factor) / factor)


def _make_quantizer(sig: Optional[int], use_float32: bool) -> Callable[[Number], Number]:
    def q(x: Number) -> Number:
        y = float(x)
        if use_float32:
            y = to_float32(y)
        if sig is not None:
            y = round_sig(y, sig)
        return y

    return q


def _deepcopy_matrix(A: Sequence[Sequence[Number]]) -> Matrix:
    return [list(map(float, row)) for row in A]


def _copy_vector(b: Sequence[Number]) -> Vector:
    return list(map(float, b))


def back_substitution(U: Sequence[Sequence[Number]], y: Sequence[Number], *, tol: float = 1e-12) -> Vector:
    """Solve an upper-triangular system Ux=y via back substitution."""
    n = len(U)
    x: Vector = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = float(y[i])
        for j in range(i + 1, n):
            s -= float(U[i][j]) * x[j]
        piv = float(U[i][i])
        if abs(piv) <= tol:
            raise ZeroDivisionError(f"Zero (or near-zero) pivot at row {i}: {piv}")
        x[i] = s / piv
    return x


def gaussian_elimination(
    A: Sequence[Sequence[Number]],
    b: Sequence[Number],
    *,
    pivot: str = "none",
    rounding_sig: Optional[int] = None,
    use_float32: bool = False,
    tol: float = 1e-12,
) -> Tuple[Matrix, Vector, List[int]]:
    """Forward elimination producing an upper-triangular matrix.

    Parameters
    - `pivot`: 'none' or 'partial'
    - `rounding_sig`: if provided, rounds intermediate arithmetic to k significant digits
    - `use_float32`: if True, quantizes arithmetic to float32 (can be combined with rounding_sig)

    Returns (U, y, perm) for Ux=y after elimination.
    """
    if pivot not in {"none", "partial"}:
        raise ValueError("pivot must be 'none' or 'partial'")

    n = len(A)
    if n == 0:
        raise ValueError("A must be non-empty")
    if any(len(row) != n for row in A):
        raise ValueError("A must be square")
    if len(b) != n:
        raise ValueError("b must have length n")

    q = _make_quantizer(rounding_sig, use_float32)

    U = _deepcopy_matrix(A)
    y = _copy_vector(b)
    perm = list(range(n))

    for k in range(n - 1):
        if pivot == "partial":
            pivot_row = max(range(k, n), key=lambda i: abs(U[i][k]))
            if pivot_row != k:
                U[k], U[pivot_row] = U[pivot_row], U[k]
                y[k], y[pivot_row] = y[pivot_row], y[k]
                perm[k], perm[pivot_row] = perm[pivot_row], perm[k]

        piv = float(U[k][k])
        if abs(piv) <= tol:
            continue

        for i in range(k + 1, n):
            m = q(U[i][k] / piv)
            U[i][k] = q(0.0)
            for j in range(k + 1, n):
                U[i][j] = q(U[i][j] - q(m * U[k][j]))
            y[i] = q(y[i] - q(m * y[k]))

    return U, y, perm


def solve_gauss(
    A: Sequence[Sequence[Number]],
    b: Sequence[Number],
    *,
    pivot: str = "none",
    rounding_sig: Optional[int] = None,
    use_float32: bool = False,
    tol: float = 1e-12,
) -> Vector:
    """Solve Ax=b using Gaussian elimination + back substitution."""
    U, y, _ = gaussian_elimination(
        A,
        b,
        pivot=pivot,
        rounding_sig=rounding_sig,
        use_float32=use_float32,
        tol=tol,
    )
    return back_substitution(U, y, tol=tol)


def rref(
    A: Sequence[Sequence[Number]],
    b: Optional[Sequence[Number]] = None,
    *,
    rounding_sig: Optional[int] = None,
    use_float32: bool = False,
    tol: float = 1e-12,
) -> Tuple[Matrix, Optional[Vector], List[int]]:
    """Compute the reduced row echelon form (Gauss-Jordan).

    If `b` is provided, transforms the augmented system consistently.
    Returns (R, rhs, pivot_cols).
    """
    q = _make_quantizer(rounding_sig, use_float32)

    M = _deepcopy_matrix(A)
    rhs = _copy_vector(b) if b is not None else None

    rows = len(M)
    cols = len(M[0]) if rows else 0
    pivot_cols: List[int] = []

    r = 0
    for c in range(cols):
        if r >= rows:
            break

        pivot_row = max(range(r, rows), key=lambda i: abs(M[i][c]))
        if abs(M[pivot_row][c]) <= tol:
            continue

        if pivot_row != r:
            M[r], M[pivot_row] = M[pivot_row], M[r]
            if rhs is not None:
                rhs[r], rhs[pivot_row] = rhs[pivot_row], rhs[r]

        piv = float(M[r][c])
        inv = q(1.0 / piv)
        for j in range(c, cols):
            M[r][j] = q(M[r][j] * inv)
        if rhs is not None:
            rhs[r] = q(rhs[r] * inv)

        for i in range(rows):
            if i == r:
                continue
            factor = float(M[i][c])
            if abs(factor) <= tol:
                M[i][c] = q(0.0)
                continue
            for j in range(c, cols):
                M[i][j] = q(M[i][j] - q(factor * M[r][j]))
            if rhs is not None:
                rhs[i] = q(rhs[i] - q(factor * rhs[r]))
            M[i][c] = q(0.0)

        pivot_cols.append(c)
        r += 1

    return M, rhs, pivot_cols


def solve_gauss_jordan(
    A: Sequence[Sequence[Number]],
    b: Sequence[Number],
    *,
    rounding_sig: Optional[int] = None,
    use_float32: bool = False,
    tol: float = 1e-12,
) -> Vector:
    """Solve Ax=b using Gauss-Jordan (RREF)."""
    M, rhs, pivot_cols = rref(A, b, rounding_sig=rounding_sig, use_float32=use_float32, tol=tol)
    if rhs is None:
        raise ValueError("b is required")

    n = len(M[0])
    x: Vector = [0.0] * n

    for i, c in enumerate(pivot_cols):
        x[c] = rhs[i]

    return x


def classify_linear_system(
    A: Sequence[Sequence[Number]],
    b: Sequence[Number],
    *,
    tol: float = 1e-12,
) -> str:
    """Classify Ax=b as 'unique', 'infinite', or 'inconsistent' using RREF."""
    M, rhs, pivot_cols = rref(A, b, tol=tol)
    if rhs is None:
        raise ValueError("b is required")

    rows = len(M)
    cols = len(M[0]) if rows else 0

    for i in range(rows):
        if all(abs(M[i][j]) <= tol for j in range(cols)) and abs(rhs[i]) > tol:
            return "inconsistent"

    rankA = len(pivot_cols)
    if rankA == cols:
        return "unique"
    return "infinite"


def alpha_system_matrix(alpha: Number) -> Matrix:
    """Matrix A(alpha) for the parameterized system in Exercise 5."""
    a = float(alpha)
    return [
        [1.0, -1.0, a],
        [-1.0, 2.0, -a],
        [a, 1.0, 1.0],
    ]


def alpha_system_rhs() -> Vector:
    """Right-hand side vector b for the parameterized system in Exercise 5."""
    return [-2.0, 3.0, 2.0]


def analyze_alpha_system(alpha: Number, *, tol: float = 1e-12) -> Tuple[str, Optional[Vector]]:
    """Return (classification, solution_if_unique) for the alpha system."""
    A = alpha_system_matrix(alpha)
    b = alpha_system_rhs()
    kind = classify_linear_system(A, b, tol=tol)
    if kind == "unique":
        return kind, solve_gauss(A, b, pivot="partial", tol=tol)
    return kind, None


def matvec(A: Sequence[Sequence[Number]], x: Sequence[Number]) -> Vector:
    """Compute A @ x for Python lists."""
    return [sum(float(aij) * float(xj) for aij, xj in zip(row, x)) for row in A]


def vec_sub(a: Sequence[Number], b: Sequence[Number]) -> Vector:
    """Compute a - b elementwise."""
    return [float(ai) - float(bi) for ai, bi in zip(a, b)]


def food_slack(A: Sequence[Sequence[Number]], x: Sequence[Number], b: Sequence[Number]) -> Vector:
    """Slack s = b - A x (positive means remaining supply)."""
    return vec_sub(b, matvec(A, x))


def food_is_enough(A: Sequence[Sequence[Number]], x: Sequence[Number], b: Sequence[Number], *, tol: float = 1e-12) -> bool:
    """Return True if A x <= b (within tolerance)."""
    s = food_slack(A, x, b)
    return all(si >= -tol for si in s)


def max_individual_increase(
    A: Sequence[Sequence[Number]],
    x: Sequence[Number],
    b: Sequence[Number],
    species_index: int,
    *,
    tol: float = 1e-12,
) -> Number:
    """Max increase Δ for one species j with others fixed so that A(x+Δe_j) <= b."""
    s = food_slack(A, x, b)
    j = species_index

    caps: List[Number] = []
    for i, row in enumerate(A):
        aij = float(row[j])
        if aij > tol:
            caps.append(float(s[i]) / aij)

    if not caps:
        return float("inf")
    return min(caps)


def extinction_max_increase(
    A: Sequence[Sequence[Number]],
    x: Sequence[Number],
    b: Sequence[Number],
    extinct_species_index: int,
    target_species_index: int,
    *,
    tol: float = 1e-12,
) -> Number:
    """Max increase for one target species after setting an extinct species population to 0."""
    x2 = list(map(float, x))
    x2[extinct_species_index] = 0.0
    return max_individual_increase(A, x2, b, target_species_index, tol=tol)


def check_solution(A: Sequence[Sequence[Number]], b: Sequence[Number], x: Sequence[Number]) -> Vector:
    """Residual r = A x - b."""
    return vec_sub(matvec(A, x), b)


def self_test() -> None:
    """Quick sanity checks; safe to run from a notebook."""
    A = [
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0],
    ]
    b = [8.0, -11.0, -3.0]
    x = solve_gauss(A, b, pivot="partial")
    r = check_solution(A, b, x)
    if max(abs(ri) for ri in r) > 1e-9:
        raise AssertionError(f"solve_gauss residual too large: {r}")

    x2 = solve_gauss_jordan(A, b)
    r2 = check_solution(A, b, x2)
    if max(abs(ri) for ri in r2) > 1e-9:
        raise AssertionError(f"solve_gauss_jordan residual too large: {r2}")
