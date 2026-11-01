from pathlib import Path
import math
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

STUDENT_NUMBER = 5
N = 100 * STUDENT_NUMBER
EXPONENT = 2 * STUDENT_NUMBER
PERIODS = (4, 8, 16, 32, 64, 128)
K_MAX = 20  # Методичка не встановлює верхню межу k; для графіків беремо 0..20


def f(t):
    return t ** EXPONENT


def fourier_term(k, T):

    omega = 2 * math.pi * k / T
    if k == 0:
        re = 2 * (float(N) ** (EXPONENT + 1)) / (EXPONENT + 1)
    else:
        integral, _ = quad(lambda x: x ** EXPONENT, 0.0, 1.0,
                           weight='cos', wvar=omega * N,
                           epsabs=1e-12, epsrel=1e-12, limit=400)
        re = 2 * (float(N) ** (EXPONENT + 1)) * integral
    im = 0.0
    return complex(re, im)


def amplitude_spectrum(coefficients):
    return np.abs(coefficients)


def calculate_spectrum(T, k_max=K_MAX):
    k_values = np.arange(k_max + 1)
    coefficients = np.array([fourier_term(int(k), T) for k in k_values])
    amplitudes = amplitude_spectrum(coefficients)
    return k_values, coefficients, amplitudes


def plot_results(results,):
    scale = 1e28
    for kind, title, filename in [
        ('real', 'Дійсна частина інтеграла Фур’є', 'real_spectra.png'),
        ('amplitude', 'Амплітудний спектр інтеграла Фур’є', 'amplitude_spectra.png'),
    ]:
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        for ax, T in zip(axes.flat, PERIODS):
            ks, coefficients, amplitudes = results[T]
            values = coefficients.real if kind == 'real' else amplitudes
            ax.plot(ks, values / scale, marker='o', markersize=3, linewidth=1.4)
            ax.axhline(0, color='gray', linewidth=.7)
            ax.set_title(f'T = {T}')
            ax.set_ylabel('Значення / 10²⁸')
            ax.set_xticks(np.arange(0, K_MAX + 1, 2))
            ax.grid(True, alpha=.25)
        for ax in axes[-1]:
            ax.set_xlabel('Номер гармоніки k')
        fig.suptitle(title + ' для f(t) = t¹⁰, N = 500', fontsize=15, y=.995)
        fig.tight_layout(rect=[0, 0, 1, .982])
        plt.show()


def main():
    print('ЛАБОРАТОРНА РОБОТА №4 — ІНТЕГРАЛЬНЕ ПЕРЕТВОРЕННЯ ФУР’Є')
    print(f'Варіант: n = {STUDENT_NUMBER}; f(t) = t^{EXPONENT}; N = {N}')
    print(f'Гармоніки: k = 0,...,{K_MAX}')
    results = {}
    f0 = 2 * float(N) ** (EXPONENT + 1) / (EXPONENT + 1)
    for T in PERIODS:
        ks, coefficients, amplitudes = calculate_spectrum(T)
        results[T] = (ks, coefficients, amplitudes)
        print(f'\nT = {T}:')
        print(f'{"k":>3} {"omega_k":>12} {"Re F":>19} {"Im F":>12} {"|F|":>19}')
        for k, coeff, amp in zip(ks, coefficients, amplitudes):
            print(f'{k:>3} {2*math.pi*k/T:>12.6f} {coeff.real:>19.8e} {coeff.imag:>12.2e} {amp:>19.8e}')
        print(f'Перевірка: |F(0)| = {f0:.8e}; Im F = 0 (парність f)')
    plot_results(results)


if __name__ == '__main__':
    main()
