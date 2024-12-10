# Дискретные преобразования

## Дискретное преобразование Фурье

Пусть:

$f: \mathbb{C} \rightarrow \mathbb{C}$

$j = \sqrt{-1}$

Тогда:

$f(t) = \int^{\infty}_{-\infty} f(w) e^{-jwt} dw$, $w = 2\pi f$, $f$ - частота, Гц

$f(w) = \int^1_0 f(t) e^{-jwt} dt$

$f_i = \sum^{N - 1}_{j = 0} f_i e^{-jwt}$ - ДПФ

---

$\int_П = f(x) = f(x - \Delta x) dx$

---

$\sum^{\infty}_{j = -\infty} f_i = f_{i - j}$

---

## Алгоритм Кули-Тьюки

$f_i = \sum^{N - 1}_{i = 0} f_i e^{-jw\frac{i}{N}} =$

$= \sum^{\frac{N}{2} - 1}_{i = 0} f_{2i} e^{-jw\frac{2i}{N}} + e^{-j\frac{i}{N}} \sum^{\frac{N}{2} - 1}_{i = 0} f_{2i + 1} e^{-jw\frac{2i}{N}} =$

$= \sum^{\frac{N}{2} - 1}_{i = 0} f_{2i} W^\frac{i}{2} + W \sum^{\frac{N}{2} - 1}_{i = 0} f_{2i + 1} W^\frac{i}{2}$

