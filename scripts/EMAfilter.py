import numpy as np

class EMAFilter:
    def __init__(self, alpha, initial_value):
        """
        EMAフィルタの初期化
        :param alpha: 平滑化係数 (0 < alpha <= 1)
        :param initial_value: 初期値
        """
        self.alpha = alpha
        self.value = initial_value

    def update(self, new_value):
        """
        新しい値でEMAフィルタを更新
        :param new_value: 新しい測定値
        :return: 更新された平滑化値
        """
        self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
