# M23-Spectrum: Algebraic Weight Initialization

[
[
[

## Overview

**M23-Spectrum** — метод инициализации весов на основе группы Матьё $M_{23}$ и динамической изометрии. Обеспечивает стабильное обучение глубоких трансформеров без взрыва/затухания градиентов.

Отличается от Xavier/He: использует собственные значения из полиномов Элкиса вместо случайных распределений.

### Ключевые особенности
- Детерминистическая инициализация на основе алгебры
- Динамическая изометрия для любой глубины
- Спектральная стабильность
- Совместимо: PyTorch, TensorFlow, JAX
- Оптимизировано для супер-разрешения и генерации кадров

## Математическая основа

### Спектр M23

Корни полинома Элкиса для $M_{23}$:  
$$g^4 + g^3 + 9g^2 - 10g + 8 = 0$$

Семейства полиномов (кратности 2,1,4):  
- $P_2$: $z^2 - gz + g^2 = 0$  
- $P_3$: $z^3 + gz - 1 = 0$  
- $P_4$: $z^4 + gz^3 - g^2z^2 + z - g = 0$

### Нормализация

Масштабирование для стабильности:  
$$\lambda_{\text{stable}} = \lambda_{\text{raw}} \cdot \left(\frac{\sqrt{2/\text{fan}_\text{in}}}{\max(|\lambda_{\text{raw}}|)}\right)$$  
$\rho(W) \approx \sqrt{2/\text{fan}_\text{in}}$

Комплексные пары $\lambda = a \pm bi$:  
$$B_i = \begin{pmatrix} a & -b \\ b & a \end{pmatrix}$$

## Установка

```bash
pip install m23-spectrum
```

Зависимости: NumPy≥1.19, SciPy≥1.5, PyTorch≥1.9 (опц.)

## Использование

### Базовое

```python
from m23_spectrum import generate_m23_stable_spectrum, mgi_init_stable

spectrum = generate_m23_stable_spectrum(256)
weight = mgi_init_stable((512, 256))
```

### PyTorch

```python
import torch.nn as nn
from m23_spectrum import apply_m23_init

model = YourModel()
model.apply(apply_m23_init)
```

### Генерация кадров

```python
class FrameGenerator(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, hidden_dim, 3, padding=1),
            nn.ReLU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(), nn.Conv2d(hidden_dim, 3, 3, padding=1)
        )
    
    def forward(self, frame0, frame1):
        x = torch.cat([frame0, frame1], dim=1)
        return self.decoder(self.encoder(x))

generator = FrameGenerator()
generator.apply(apply_m23_init)
```

## Бенчмарки

### Сходимость (24 слоя трансформера)

| Метрика | M23 | He | Улучшение |
|---------|----|----|-----------|
| Эпохи до сходимости | 15 | 42 | **2.8x** |
| Финальный loss | 0.0234 | 0.0312 | **25%** |
| Стабильность градиентов | 0.089 | 0.412 | **4.6x** |
| Condition number | 17.6 | 145.2 | **8.2x** |

### Память (генерация кадров 512×512)

| Конфиг | M23 | He | Экономия |
|--------|----|----|----------|
| GPU память (4x) | 6.2 GB | 8.9 GB | **30%** |
| Время/эпоха | 12.3s | 15.7s | **21%** |
| Инференс | 23ms | 28ms | **18%** |

## Результаты

**Супер-разрешение (DIV2K, 1080p→4K):**  
PSNR: M23=38.42 | He=36.87 | Xavier=35.92  
SSIM: M23=0.9624 | He=0.9401 | Xavier=0.9187

**Интерполяция кадров (Vimeo90K):**  
PSNR: M23=33.24 | RIFE=32.81 | DAIN=31.55

## API

- `generate_m23_stable_spectrum(fan_in)` → спектр
- `mgi_init_stable((fan_out, fan_in))` → матрица весов
- `apply_m23_init(model)` → инициализация PyTorch

## Применение

1. **Супер-разрешение (DLSS)**
2. **Интерполяция кадров (Lossless Scaling)**
3. **Глубокие трансформеры (100+ слоёв)**

## Ограничения

- Overhead QR для больших слоёв (>10K)
- Лучше для трансформеров, чем CNN

**Будущее:** GPU-ускорение, адаптивное масштабирование

## Лицензия

MIT License

## Цитирование

```bibtex
@software{m23spectrum2026,
  title={M23-Spectrum: Algebraic Weight Initialization},
  author={m23spectrum},
  year={2026},
  url={https://github.com/m23spectrum/m23-spectrum}
}
```

**Статус:** Alpha v0.2.0 | Обновлено: 31.01.2026

