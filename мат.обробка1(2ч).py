import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

n = 50
a = -np.pi
b = np.pi

# Довжина інтервалу
T = b - a

# Основна кутова частота
w0 = 2 * np.pi / T


# 1. Точне обчислення заданої функції

def f(x):
    return (x ** n) * np.exp(-(x ** 2) / n)


# 2. Обчислення коефіцієнтів Фур'є

def a_0():

    integral, _ = quad(
        lambda x: f(x),
        a,
        b
    )

    return (2 / T) * integral


def a_k(k):

    integral, _ = quad(
        lambda x: f(x) * np.cos(w0 * k * x),
        a,
        b
    )

    return (2 / T) * integral


def b_k(k):
    return 0.0


def calculate_coefficients(N):

    a_values = np.zeros(N + 1)
    b_values = np.zeros(N + 1)

    a_values[0] = a_0()

    for k in range(1, N + 1):
        a_values[k] = a_k(k)
        b_values[k] = b_k(k)

    return a_values, b_values

# 3. Наближення функції рядом Фур'є

def fourier_approximation(x, N, a_values, b_values):


    result = np.zeros_like(
        np.asarray(x, dtype=float)
    )

    # Постійна складова
    result += a_values[0] / 2

    # Гармоніки
    for k in range(1, N + 1):

        result += (
            a_values[k] * np.cos(w0 * k * x)
            +
            b_values[k] * np.sin(w0 * k * x)
        )

    return result


# Графік точної функції та наближення

def plot_approximation(N, a_values, b_values):

    x = np.linspace(a, b, 2000)

    y_exact = f(x)

    y_fourier = fourier_approximation(
        x,
        N,
        a_values,
        b_values
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        x,
        y_exact,
        label="Точна функція f(x)",
        linewidth=2
    )

    plt.plot(
        x,
        y_fourier,
        "--",
        label=f"Наближення рядом Фур'є, N = {N}",
        linewidth=2
    )

    plt.plot(
        x,
        y_fourier,
        "--",
        label=f"Наближення рядом Фур'є, N = {N}",
        linewidth=2
    )

    plt.title(
        "Наближення функції "
        "f(x) = x^50 exp(-x²/50) рядом Фур'є"
    )

# 4. Побудова графіків гармонік

def plot_harmonics(N, a_values, b_values):

    x = np.linspace(a, b, 2000)

    plt.figure(figsize=(13, 7))

    #  k = 0
    harmonic_0 = np.full_like(
        x,
        a_values[0] / 2
    )

    plt.plot(
        x,
        harmonic_0,
        label="k = 0"
    )

    # Гармоніки від 1 до N
    for k in range(1, N + 1):

        harmonic = (
            a_values[k] *
            np.cos(w0 * k * x)
            +
            b_values[k] *
            np.sin(w0 * k * x)
        )

        plt.plot(
            x,
            harmonic,
            label=f"k = {k}"
        )

    plt.title(
        "Гармоніки тригонометричного ряду Фур'є"
    )

    plt.xlabel("x")
    plt.ylabel("Амплітуда")

    plt.grid(True)

    plt.legend(
        loc="upper right",
        ncol=2
    )

    plt.tight_layout()
    plt.show()



# 4. Частотна область
# Коефіцієнти a_k та b_k

def plot_frequency_domain(
        N,
        a_values,
        b_values
):

    k_values = np.arange(N + 1)

    # Коефіцієнти a_k

    plt.figure(figsize=(10, 5))

    plt.stem(
        k_values,
        a_values
    )

    plt.title(
        "Частотна область: коефіцієнти a_k"
    )

    plt.xlabel("Номер гармоніки k")
    plt.ylabel("a_k")

    plt.xticks(k_values)

    plt.grid(True)

    plt.tight_layout()
    plt.show()


    # Коефіцієнти b_k

    plt.figure(figsize=(10, 5))

    plt.stem(
        k_values,
        b_values
    )

    plt.title(
        "Частотна область: коефіцієнти b_k"
    )

    plt.xlabel("Номер гармоніки k")
    plt.ylabel("b_k")

    plt.xticks(k_values)

    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 5. Оцінка відносної похибки

def relative_error(
        N,
        a_values,
        b_values
):


    x = np.linspace(
        a,
        b,
        10000
    )

    exact = f(x)

    approximation = fourier_approximation(
        x,
        N,
        a_values,
        b_values
    )

    # Інтеграл квадрата похибки
    numerator = np.trapezoid(
        (exact - approximation) ** 2,
        x
    )

    # Інтеграл квадрата точної функції
    denominator = np.trapezoid(
        exact ** 2,
        x
    )

    error = (
        np.sqrt(numerator)
        /
        np.sqrt(denominator)
    ) * 100

    return error


# 6. Збереження результатів у файл

def save_results(
        N,
        a_values,
        b_values,
        error
):

    filename = "results.txt"

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "ЛАБОРАТОРНА РОБОТА №1\n"
        )

        file.write(
            "Наближення функції "
            "тригонометричним рядом Фур'є\n\n"
        )

        file.write(
            "Номер у списку: n = 5\n"
        )


        file.write(
            "Інтервал: [-pi, pi]\n\n"
        )

        file.write(
            f"Порядок наближення N = {N}\n\n"
        )

        file.write(
            "Коефіцієнти Фур'є:\n\n"
        )

        # a0
        file.write(
            f"a0 = {a_values[0]:.10f}\n\n"
        )

        # Решта коефіцієнтів
        for k in range(1, N + 1):

            file.write(
                f"k = {k:2d}: "
                f"a{k} = {a_values[k]: .10f}, "
                f"b{k} = {b_values[k]: .10f}\n"
            )

        file.write("\n")

        file.write(
            f"Відносна похибка = "
            f"{error:.10f}%\n"
        )

    print(
        f"\nРезультати збережено "
        f"у файл {filename}"
    )



def main():

    N = 10


    print(
        "Інтервал: [-pi, pi]"
    )

    print(
        f"T = {T}"
    )

    print(
        f"w0 = {w0}"
    )

    print(
        f"\nПорядок наближення N = {N}"
    )

    a_values, b_values = (
        calculate_coefficients(N)
    )

    print(
        "\nКоефіцієнти ряду Фур'є:"
    )

    print(
        f"a0 = {a_values[0]: .10f}"
    )

    for k in range(1, N + 1):

        print(
            f"k = {k:2d}: "
            f"a{k} = {a_values[k]: .10f}, "
            f"b{k} = {b_values[k]: .10f}"
        )


    error = relative_error(
        N,
        a_values,
        b_values
    )

    print(
        f"\nВідносна похибка: "
        f"{error:.10f}%"
    )

    plot_approximation(
        N,
        a_values,
        b_values
    )
    plot_harmonics(
        N,
        a_values,
        b_values
    )

    plot_frequency_domain(
        N,
        a_values,
        b_values
    )

    save_results(
        N,
        a_values,
        b_values,
        error
    )

if __name__ == "__main__":
    main()