import math
import time
import numpy as np
import matplotlib.pyplot as plt

# Частина I - Пряме дискретне перетворення Фур'є
# Частина II - Відтворення аналогового сигналу
# Частина III - Зворотне дискретне перетворення Фур'є

STUDENT_NUMBER = 5

# Налаштування графіків
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.grid"] = True

# ЧАСТИНА I

def fourier_term(f_i, k, i, N):
    angle = 2 * math.pi * k * i / N

    A = (f_i / N) * math.cos(angle)
    B = -(f_i / N) * math.sin(angle)

    return complex(A, B)


def fourier_coefficient(signal, k):
    N = len(signal)

    C_k = 0j

    adds = 0
    mult = 0
    for i in range(N):
        mult += 2
        if (i > 0):
            adds += 2
        C_k += fourier_term(signal[i], k, i, N)

    return C_k, adds, mult

def calculate_dft(signal):
    N = len(signal)

    coefficients = []
    adds_count = 0
    mult_count = 0

    for k in range(N):
        C_k, adds, mult = fourier_coefficient(signal, k)
        mult_count += mult
        adds_count += adds
        coefficients.append(C_k)

    return np.array(coefficients), adds_count, mult_count


def plot_spectra(coefficients):
    N = len(coefficients)

    k_values = np.arange(N)

    amplitudes = np.abs(coefficients)
    phases = np.angle(coefficients)

    # При нульовому коефіцієнті фаза не визначена.
    phases[amplitudes < 1e-10] = np.nan

    # Графік 1. Амплітудний спектр
    plt.figure(figsize=(10, 5))

    plt.stem(k_values, amplitudes)

    plt.title("Частина I. Амплітудний спектр ДПФ")
    plt.xlabel("Номер гармоніки k")
    plt.ylabel("|C_k|")

    plt.xticks(k_values)
    plt.tight_layout()

    # Графік 2. Фазовий спектр
    plt.figure(figsize=(10, 5))

    plt.stem(k_values, phases)

    plt.title("Частина I. Фазовий спектр ДПФ")
    plt.xlabel("Номер гармоніки k")
    plt.ylabel("Фаза, рад")

    plt.xticks(k_values)
    plt.tight_layout()

def count_operations(N):
    total_terms = N * N

    multiplications = 4 * total_terms
    divisions = total_terms
    additions = 2 * total_terms

    return multiplications, additions, divisions

def part_one():
    print("ЧАСТИНА I. ДИСКРЕТНЕ ПЕРЕТВОРЕННЯ ФУР'Є")

    n = STUDENT_NUMBER
    N = 10 + n

    print(f"\nНомер у журналі: {n}")
    print(f"N = 10 + {n} = {N}")

    # Генерація довільного вектора з N значень
    np.random.seed(5)

    signal = np.random.randint(0, 10, size=N)

    print("\nВхідний вектор:")
    print(signal)

    # Вимірювання часу ДПФ
    start_time = time.perf_counter()

    coefficients, adds, mult = calculate_dft(signal)
    print(f"adds = {adds} mult {mult}")

    end_time = time.perf_counter()

    calculation_time = end_time - start_time

    print("\nКомплексні коефіцієнти Фур'є:")

    print(
        f"{'k':<5}"
        f"{'A_k':<15}"
        f"{'B_k':<15}"
        f"{'|C_k|':<15}"
        f"{'Фаза (рад)':<15}"
    )

    for k, C_k in enumerate(coefficients):

        A_k = C_k.real
        B_k = C_k.imag

        amplitude = abs(C_k)

        if amplitude < 1e-10:
            phase_text = "не визначена"
        else:
            phase_text = f"{np.angle(C_k):.6f}"

        print(
            f"{k:<5}"
            f"{A_k:<15.6f}"
            f"{B_k:<15.6f}"
            f"{amplitude:<15.6f}"
            f"{phase_text:<15}"
        )

    # Підрахунок операцій
    # mult, add, div = count_operations(N)

    # print(f"adds {adds} mul = {mult}")

    print("\nОцінювання обчислень:")

    print(f"Час виконання ДПФ: {calculation_time:.8f} с")
    # print(f"Кількість множень: {mult}")
    # print(f"Кількість додавань: {add}")
    # print(f"Кількість ділень: {div}")

    # Побудова двох спектрів
    plot_spectra(coefficients)

    return signal, coefficients

# ЧАСТИНА II
# ВІДТВОРЕННЯ АНАЛОГОВОГО СИГНАЛУ

def generate_binary_signal(student_number):

    N = 96 + student_number

    # Число в 8-бітному двійковому вигляді
    binary = format(N, "08b")

    # Восьмий розряд з правого боку - старший біт
    bits = list(binary)

    if student_number % 2 == 1:
        bits[0] = "1"
    else:
        bits[0] = "0"

    modified_binary = "".join(bits)

    signal = np.array(
        [int(bit) for bit in modified_binary],
        dtype=float
    )

    return N, binary, modified_binary, signal


def reconstruct_signal(t, coefficients, Tc=1.0):

    C0 = coefficients[0].real
    C4 = coefficients[4].real

    # Постійна складова
    result = np.full_like(t, C0, dtype=float)

    # Перша, друга і третя гармоніки
    for k in range(1, 4):

        amplitude = abs(coefficients[k])
        phase = np.angle(coefficients[k])

        result += (
            2 * amplitude *
            np.cos(2 * np.pi * k * t / Tc + phase)
        )

    # Четверта гармоніка
    result += C4 * np.cos(8 * np.pi * t / Tc)

    return result


# Побудова графіка відновленого сигналу

def plot_reconstructed_signal(signal, coefficients):

    Tc = 1.0

    # 1000 точок для неперервної кривої
    t = np.linspace(0, Tc, 1000)

    reconstructed = reconstruct_signal(
        t, coefficients, Tc
    )

    # Координати початкових 8 відліків
    sample_t = np.arange(8) * Tc / 8

    plt.figure(figsize=(11, 5))

    plt.plot(
        t,
        reconstructed,
        label="Відтворений аналоговий сигнал",
        linewidth=2
    )

    plt.scatter(
        sample_t,
        signal,
        label="Початкові 8 відліків",
        s=65,
        zorder=5
    )

    plt.title(
        "Частина II. Відтворення аналогового сигналу"
    )

    plt.xlabel("Нормований час t / Tc")
    plt.ylabel("s(t)")

    plt.xlim(0, Tc)
    plt.xticks(np.arange(0, 1.01, 0.125))

    plt.legend()
    plt.tight_layout()

def part_two():

    print("ЧАСТИНА II. ВІДТВОРЕННЯ АНАЛОГОВОГО СИГНАЛУ")

    N, binary, modified_binary, signal = (
        generate_binary_signal(STUDENT_NUMBER)
    )

    print(f"\nN = 96 + {STUDENT_NUMBER} = {N}")

    print(f"Початковий двійковий запис: {binary}")
    print(f"Після зміни 8-го розряду: {modified_binary}")

    print("\nВхідні 8 відліків:")

    for i, value in enumerate(signal):
        print(f"s({i}Tδ) = {int(value)}")

    # Обчислення коефіцієнтів ДПФ
    coefficients, adds, mult = calculate_dft(signal)

    print("\nКомплексні коефіцієнти C_k:")

    print(
        f"{'k':<5}"
        f"{'Re(C_k)':<15}"
        f"{'Im(C_k)':<15}"
        f"{'|C_k|':<15}"
        f"{'arg(C_k)':<15}"
    )

    for k in range(5):

        C_k = coefficients[k]

        amplitude = abs(C_k)

        if amplitude < 1e-10:
            phase_text = "не визначена"
        else:
            phase_text = f"{np.angle(C_k):.6f}"

        print(
            f"{k:<5}"
            f"{C_k.real:<15.6f}"
            f"{C_k.imag:<15.6f}"
            f"{amplitude:<15.6f}"
            f"{phase_text:<15}"
        )

    # Виведення аналітичного виразу s(t)
    print("\nАналітичний вираз сигналу s(t):")
    print("Для нормованого періоду Tc = 1:")

    expression = f"s(t) = {coefficients[0].real:.6f}"

    for k in range(1, 4):

        amplitude = 2 * abs(coefficients[k])
        phase = np.angle(coefficients[k])

        expression += (
            f" + {amplitude:.6f}"
            f"*cos({2*k}*pi*t {phase:+.6f})"
        )

    expression += (
        f" {coefficients[4].real:+.6f}"
        f"*cos(8*pi*t)"
    )

    print(expression)

    # Додаткова таблиця на 16 проміжках
    t_values = np.linspace(0, 1, 17)

    values = reconstruct_signal(
        t_values, coefficients
    )

    print("\nЗначення відтвореного сигналу:")
    print(f"{'t/Tc':<15}{'s(t)':<15}")

    for t, value in zip(t_values, values):
        print(f"{t:<15.6f}{value:<15.6f}")

    # Графік відновленого сигналу
    plot_reconstructed_signal(signal, coefficients)

    return signal, coefficients


def inverse_dft(coefficients):

    N = len(coefficients)

    result = []

    for n in range(N):

        value = 0.0

        for k in range(N):

            A_k = coefficients[k].real
            B_k = coefficients[k].imag

            angle = 2 * math.pi * k * n / N

            value += (
                A_k * math.cos(angle)
                - B_k * math.sin(angle)
            )

        # Прибираємо тільки похибки округлення
        if abs(value) < 1e-10:
            value = 0.0

        result.append(value)

    return np.array(result)


def analytical_samples(coefficients):

    C0 = coefficients[0].real
    C4 = coefficients[4].real

    # Нульовий відлік
    s0 = C0 + C4

    for k in range(1, 4):
        s0 += 2 * coefficients[k].real

    # Перший відлік
    s1 = C0 - C4

    terms_s1 = []

    for k in range(1, 4):

        A_k = coefficients[k].real
        B_k = coefficients[k].imag

        angle = math.pi * k / 4

        term = 2 * (
            A_k * math.cos(angle)
            - B_k * math.sin(angle)
        )

        terms_s1.append(term)
        s1 += term

    print("\nАналітичне обчислення перших двох відліків")

    print("\nДля n = 0:")
    print(
        "s(0) = C0 + 2Re(C1) + 2Re(C2)"
        " + 2Re(C3) + C4"
    )

    print(
        f"s(0) = {C0:.6f}"
        f" + 2*({coefficients[1].real:.6f})"
        f" + 2*({coefficients[2].real:.6f})"
        f" + 2*({coefficients[3].real:.6f})"
        f" + ({C4:.6f})"
    )

    print(f"s(0) = {s0:.6f}")

    print("\nДля n = 1:")
    print(
        "s(Tδ) = C0 + 2Re(C1*exp(j*pi/4))"
        " + 2Re(C2*exp(j*pi/2))"
        " + 2Re(C3*exp(j*3*pi/4)) - C4"
    )

    print(
        f"s(Tδ) = {C0:.6f}"
        + "".join(f" + ({v:.6f})" for v in terms_s1)
        + f" - ({C4:.6f})"
    )

    print(f"s(Tδ) = {s1:.6f}")

    return s0, s1

def part_three(original_signal, coefficients):

    print("ЧАСТИНА III. ЗВОРОТНЕ ДИСКРЕТНЕ ПЕРЕТВОРЕННЯ")

    restored_signal = inverse_dft(coefficients)

    print("\nПорівняння початкових і відновлених відліків:")

    print(
        f"{'n':<5}"
        f"{'Початковий':<18}"
        f"{'Відновлений':<18}"
        f"{'Похибка':<15}"
    )

    for n in range(len(original_signal)):

        original = original_signal[n]
        restored = restored_signal[n]

        error = abs(original - restored)

        print(
            f"{n:<5}"
            f"{original:<18.6f}"
            f"{restored:<18.6f}"
            f"{error:<15.10f}"
        )

    # Аналітичні обчислення для n = 0, 1
    s0, s1 = analytical_samples(coefficients)

    # Перевірка правильності
    max_error = np.max(
        np.abs(original_signal - restored_signal)
    )

    print("\nПеревірка правильності ЗДПФ:")

    print(f"Максимальна похибка: {max_error:.12f}")

    if np.allclose(
        original_signal,
        restored_signal,
        atol=1e-10
    ):
        print("Результат: усі 8 відліків відновлено правильно.")
    else:
        print("Результат: є розбіжності у відновлених відліках.")

    return restored_signal

def main():

    print("ЛАБОРАТОРНА РОБОТА №2")
    print("ДИСКРЕТНЕ ПЕРЕТВОРЕННЯ ФУР'Є")
    print("Номер у журналі:", STUDENT_NUMBER)

    # ЧАСТИНА I
    signal_part1, coefficients_part1 = part_one()

    # ЧАСТИНА II
    signal_part2, coefficients_part2 = part_two()

    # ЧАСТИНА III
    restored_signal = part_three(
        signal_part2,
        coefficients_part2
    )

    plt.show()


if __name__ == "__main__":
    main()
