import cmath
import math
import statistics
import time

import numpy as np

STUDENT_NUMBER = 5
N = 10 + STUDENT_NUMBER


def calculate_dft(signal):
    n = len(signal)
    coefficients = []
    adds = 0
    mults = 0
    for k in range(n):
        coefficient = 0j
        for i, value in enumerate(signal):
            angle = 2 * math.pi * k * i / n
            a = (value / n) * math.cos(angle)
            b = -(value / n) * math.sin(angle)
            coefficient += complex(a, b)
            mults += 2
            if i > 0:
                adds += 2
        coefficients.append(coefficient)
    return np.array(coefficients), adds, mults


def smallest_factor(n):
    for f in range(2, math.isqrt(n) + 1):
        if n % f == 0:
            return f
    return n


def fft_mixed_radix(signal):

    def transform(x):
        n = len(x)
        if n == 1:
            return [complex(x[0])], 0, 0
        radix = smallest_factor(n)
        if radix == n:
            out = []
            adds = 0
            mults = 0
            for k in range(n):
                s = 0j
                for i, value in enumerate(x):
                    angle = -2 * math.pi * k * i / n
                    w = complex(math.cos(angle), math.sin(angle))
                    s += value * w
                    if abs(value.imag) < 1e-14:
                        mults += 2  # дійсне число * комплексне
                    else:
                        mults += 4  # комплексне * комплексне
                        adds += 2  # суми дійсних добутків
                    if i > 0:
                        adds += 2  # комплексне додавання
                out.append(s)
            return out, adds, mults

        m = n // radix
        groups = []
        adds = 0
        mults = 0
        for j in range(radix):
            values, a, b = transform(x[j::radix])
            groups.append(values)
            adds += a
            mults += b

        out = []
        for k in range(n):
            s = 0j
            for j in range(radix):
                angle = -2 * math.pi * j * k / n
                w = complex(math.cos(angle), math.sin(angle))
                s += groups[j][k % m] * w
                mults += 4
                adds += 2  # комплексне множення
                if j > 0:
                    adds += 2  # комплексне додавання
            out.append(s)
        return out, adds, mults

    raw, adds, mults = transform([complex(v) for v in signal])
    return np.array(raw, dtype=complex) / len(signal), adds, mults


def benchmark(func, signal, repeats=100):
    func(signal)  # прогрів
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(signal)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    np.random.seed(5)
    signal = np.random.randint(0, 10, size=N)
    dft, dft_adds, dft_mults = calculate_dft(signal)
    fft, fft_adds, fft_mults = fft_mixed_radix(signal)

    dft_time = benchmark(calculate_dft, signal)
    fft_time = benchmark(fft_mixed_radix, signal)

    print('ЛАБОРАТОРНА РОБОТА №3 — ШВИДКЕ ПЕРЕТВОРЕННЯ ФУР’Є')
    print(f'Варіант: {STUDENT_NUMBER}; N = {N}')
    print('Вхідний вектор:', signal)
    print('\nКоефіцієнти ДПФ та ШПФ:')
    print(f"{'k':<4}{'ДПФ (Re)':>13}{'ДПФ (Im)':>13}{'ШПФ (Re)':>13}{'ШПФ (Im)':>13}")
    for k in range(N):
        print(f'{k:<4}{dft[k].real:>13.6f}{dft[k].imag:>13.6f}'
              f'{fft[k].real:>13.6f}{fft[k].imag:>13.6f}')

    error = np.max(np.abs(dft - fft))
    print('\nПОРІВНЯННЯ')
    print(f"{'Показник':<29}{'ДПФ':>15}{'ШПФ':>15}")
    print(f"{'Час (медіана 100 запусків), с':<29}{dft_time:>15.8f}{fft_time:>15.8f}")
    print(f"{'Множення (умовна оцінка)':<29}{dft_mults:>15}{fft_mults:>15}")
    print(f"{'Додавання (умовна оцінка)':<29}{dft_adds:>15}{fft_adds:>15}")
    print(f'Максимальна різниця коефіцієнтів: {error:.3e}')
    print('Збіг коефіцієнтів:', np.allclose(dft, fft, atol=1e-10))
    print('Важливо: оцінка операцій залежить від реалізації; час — від комп’ютера.')


if __name__ == '__main__':
    main()
