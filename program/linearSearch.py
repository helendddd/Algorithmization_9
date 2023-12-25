#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import timeit
import numpy as np
import matplotlib.pyplot as plt


def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1


def find_coeffs_line(xs, ys):
    sx = sum(xs)
    stime = sum(ys)
    sx2 = sum(i**2 for i in xs)
    sxtime = sum(i*j for i, j in zip(xs, ys))
    n = len(xs)
    matrixx = [[sx2, sx], [sx, n]]
    matrixy = [[sxtime], [stime]]
    x = np.linalg.solve(matrixx, matrixy)
    return x[0][0], x[1][0]


def generate_data(size):
    arr = list(range(size))
    target_avg = arr[size // 2]  # Искомый элемент для среднего случая
    target_worst = size  # Значение, которого нет в массиве для худшего случая
    return arr, target_avg, target_worst


def measure_time(func, *args):
    return timeit.timeit(lambda: func(*args), number=100) / 100


def analyze_linear_search(start, end, step):
    sizes = list(range(start, end + 1, step))
    times_avg = []
    times_worst = []

    for size in sizes:
        arr, target_avg, target_worst = generate_data(size)

        time_avg = measure_time(linear_search, arr, target_avg)
        time_worst = measure_time(linear_search, arr, target_worst)

        times_avg.append(time_avg)
        times_worst.append(time_worst)

    return sizes, times_avg, times_worst


def plot_results(sizes, times_avg, times_worst):
    plt.scatter(sizes, times_avg, label='Средний случай', s=10)
    plt.scatter(sizes, times_worst, label='Худший случай', s=10)

    a_avg, b_avg = find_coeffs_line(sizes, times_avg)
    a_worst, b_worst = find_coeffs_line(sizes, times_worst)

    y_fit_avg = a_avg * np.array(sizes) + b_avg
    y_fit_worst = a_worst * np.array(sizes) + b_worst

    plt.plot(sizes, y_fit_avg, label=f'Для среднего: \
    {a_avg:.9f} * x + {b_avg:.9f}', color='red')
    plt.plot(sizes, y_fit_worst, label=f'Для худшего: \
    {a_worst:.9f} * x + {b_worst:.9f}', color='blue')

    plt.xlabel('Размер массива')
    plt.ylabel('Время выполнения (секунды)')
    plt.title('Анализ времени выполнения линейного поиска')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    start = 100
    end = 10000
    step = 50

    sizes, times_avg, times_worst = analyze_linear_search(start, end, step)
    plot_results(sizes, times_avg, times_worst)
