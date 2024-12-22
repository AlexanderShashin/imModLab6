import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

def holt_winters_forecasting(
    data: List[float],
    alpha: float,
    beta: float,
    gamma: float,
    seasons: int,
    forecast_periods: int
) -> Tuple[List[float], List[float], List[float], List[float]]:

    n = len(data)
    if n < 2 * seasons:
        raise ValueError("Недостаточно данных для сезонной декомпозиции. Требуется как минимум два цикла данных.")

    # Инициализация компонентов
    level = [0] * n
    trend = [0] * n
    season = [0] * n
    forecast = [0] * (n + forecast_periods)

    # Начальные оценки
    level[0] = np.mean(data[:seasons])
    trend[0] = (np.mean(data[seasons:2*seasons]) - np.mean(data[:seasons])) / seasons
    for i in range(seasons):
        season[i] = data[i] - level[0]

    # Рекурсивные вычисления
    for t in range(seasons, n):
        level[t] = alpha * (data[t] - season[t - seasons]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        season[t] = gamma * (data[t] - level[t]) + (1 - gamma) * season[t - seasons]
        forecast[t] = level[t] + trend[t] + season[t - seasons]

    # Прогнозирование будущих периодов
    for t in range(n, n + forecast_periods):
        forecast[t] = level[-1] + (t - n + 1) * trend[-1] + season[t - seasons]

    return level, trend, season, forecast[-forecast_periods:]


if __name__ == "__main__":

    mock_data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                 115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140]

    alpha, beta, gamma = 0.2, 0.1, 0.3  # Параметры сглаживания
    seasons = 4  # Сезонность по кварталам
    forecast_periods = 4  # Прогноз на следующие 4 периода

    level, trend, season, forecast = holt_winters_forecasting(
        mock_data, alpha, beta, gamma, seasons, forecast_periods
    )

    print("Компоненты уровня:", level)
    print("Компоненты тренда:", trend)
    print("Компоненты сезонности:", season[:seasons])  # Печать первого цикла сезонности
    print("Прогноз:", forecast)

    plt.figure(figsize=(12, 6))

    plt.plot(range(len(mock_data)), mock_data, label="Исходные данные", marker="o")

    # Уровень
    plt.plot(range(len(level)), level, label="Уровень", linestyle="--")

    # Прогноз
    plt.plot(range(len(mock_data), len(mock_data) + forecast_periods), forecast, label="Прогноз", marker="o", linestyle="--")

    # Тренд
    plt.plot(range(len(trend)), trend, label="Тренд", linestyle="--")

    plt.title("Модель Хольта-Уинтерса")
    plt.xlabel("Периоды")
    plt.ylabel("Значения")
    plt.legend()
    plt.grid()
    plt.show()
