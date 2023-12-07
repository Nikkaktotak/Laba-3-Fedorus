import numpy as np
import matplotlib.pyplot as plt


def simulate_absorbing_markov_chain(P, p0, T):
    """
    Симулирует поглощающий цепь Маркова с заданными матрицей переходов P, вектором начальных вероятностей p0 и длиной реализации T.

    Args:
        P: Матрица переходов цепи Маркова.
        p0: Вектор начальных вероятностей.
        T: Длина реализации.

    Returns:
        Список состояний цепи Маркова.
    """

    x = np.random.choice(2, size=T, p=p0)
    while x[-1] != 1:
        x = np.append(x, np.random.choice(2, p=P[x[-1], :]))
    return x


def visualize_absorbing_markov_chain(x):
    """
    Визуализирует реализацию поглощающего цепи Маркова.

    Args:
        x: Список состояний цепи Маркова.

    Returns:
        None.
    """
    plt.figure(figsize=(15, 4))
    plt.plot(x)
    plt.show()


def plot_characteristics(theoretical_characteristics, experimental_characteristics):
    """
  Візуалізує теоретичні та експериментальні характеристики марковського ланцюга.

  Аргументи:
    theoretical_characteristics: Теоретичний набір характеристик.
    experimental_characteristics: Експериментальний набір характеристик.
  """

    plt.figure(figsize=(10, 4))
    # Осьова система координат.
    plt.xlabel("Характеристика")
    plt.ylabel("Значення")

    # Теоретичні характеристики.
    plt.plot(["Очікуваний час перебування в стані A", "Час поглинання в стані A", "Ймовірність поглинання в стані A"],
             theoretical_characteristics, label="Теоретичні")

    # Експериментальні характеристики.
    plt.plot(["Очікуваний час перебування в стані A", "Час поглинання в стані A", "Ймовірність поглинання в стані A"],
             experimental_characteristics, label="Експериментальні")

    # Легенда.
    plt.legend()

    # Оформлення графіка.
    plt.title("Теоретичні та експериментальні характеристики марковського ланцюга")
    plt.grid(True)

    # Відображення графіку.
    plt.show()


# Задані перехідні ймовірності


if __name__ == "__main__":
    P = np.array([[0.9, 0.1], [0, 1]])
    p0 = np.array([0.5, 0.5])
    T = 110

    x = simulate_absorbing_markov_chain(P, p0, T)
    visualize_absorbing_markov_chain(x)
    # Теоретичні характеристики.
    theoretical_characteristics = [10 / 7, 10 / 7, 7 / 14]

    # Експериментальні характеристики.
    experimental_characteristics = [12.5 / 11, 12.5 / 11, 7 / 14]

    # Візуалізація.
    plot_characteristics(theoretical_characteristics, experimental_characteristics)

    P = [[0.2, 0.8], [0.6, 0.4]]

    # Початкові ймовірності
    p_0 = [0.5, 0.5]

    # Довжина реалізації
    T = 10

    # Кількість реалізацій
    N = 1000

    # Генерація реалізацій
    plt.figure(figsize=(15, 4))
    realizations = []
    for i in range(N):
        realization = [np.random.choice(2, p=P[i if i < len(P) else 0]) for i in range(T)]
        realizations.append(realization)

    # Конвертація реалізацій у NumPy array
    realizations_array = np.array(realizations)

    # Візуалізація
    plt.plot([i for i in range(T)], [np.mean(realizations_array.ravel()[i:i + N]) for i in range(T)])
    plt.xlabel("Крок")
    plt.ylabel("Значення стану")
    plt.show()


    # Параметри моделі
    n_states = 3
    p_ij = np.array([[0.5, 0.2, 0.3], [0.1, 0.7, 0.2], [0.4, 0.1, 0.5]])

    # Теоретичні характеристики
    P = p_ij
    T_i = []
    for i in range(n_states):
        T_i.append(np.sum(P[i, :]))
    P_s = np.linalg.matrix_power(P, n_states)

    # Експериментальні характеристики
    n_trials = 10000
    X_i = np.zeros(n_trials, dtype=int)
    for i in range(n_trials):
        state = 0
        for j in range(n_states):
            if np.random.rand() < p_ij[state, j]:
                state = j
        X_i[i] = state
    T_e_i = np.mean(X_i == i)
    P_e = np.array([np.mean(X_i == 0), np.mean(X_i == 1), np.mean(X_i == 2)])

    # Візуалізація
    plt.matshow(P)
    plt.title("Теоретична матриця перехідних ймовірностей")
    plt.show()

    plt.hist(X_i, bins=n_states)
    plt.title("Експериментальний розподіл частот часу перебування в стані 0")
    plt.show()

    plt.matshow(P_s)
    plt.title("Стаціонарний режим")
    plt.show()
