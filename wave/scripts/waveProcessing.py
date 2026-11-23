# waveProcessing.py
import numpy as np
import matplotlib.pyplot as plt


def read(filename):
    samples = []
    duration = None
    count = None
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("- Wave Lab"):
                continue
            if line.startswith("#"):
                for part in line[1:].split(","):
                    part = part.strip()
                    if part.startswith("duration"):
                        duration = float(part.split()[1])
                    elif part.startswith("samples"):
                        count = int(part.split()[1])
                continue
            samples.append(int(line))
    return np.array(samples), duration, count


dataPath = "../data/"
heights = [20, 40, 60, 80, 100]
adcMeans = []

for h in heights:
    samples, duration, count = read(dataPath + "calibration/%dmm.txt" % h)
    adcMeans.append(np.mean(samples))
    print(f"{h} мм: среднее {np.mean(samples):.1f}, "
          f"частота {count/duration:.0f} Гц")

adcMeans = np.array(adcMeans)

levels = np.array(heights) / 1000
fit = np.polyfit(adcMeans, levels, 2)

fitted = np.polyval(fit, adcMeans)
for h, a, l in zip(heights, adcMeans, fitted):
    print(f"{h} мм: отсчёт {a:.1f} → уровень {l*1000:.1f} мм, "
          f"отклонение {(l*1000 - h):+.1f} мм")

minAdc = adcMeans.min()
maxAdc = adcMeans.max()

x = np.linspace(minAdc, maxAdc, 200)
y = np.polyval(fit, x) * 1000

plt.figure()
plt.plot(adcMeans, heights, "o", label="калибровка")
plt.plot(x, y, "-", label="многочлен 3-й степени")
plt.xlabel("отсчёт АЦП")
plt.ylabel("уровень воды, мм")
plt.title("Калибровка датчика уровня")
plt.legend()
plt.grid(True)
#plt.savefig("../../height-calibration.png", dpi=150)
plt.show()

samples, duration, count = read(dataPath + "wave40mm.txt")
samples = np.clip(samples, minAdc, maxAdc)
level_mm = np.polyval(fit, samples) * 1000
print(f"Опыт 40 мм: уровень от {level_mm.min():.1f} до {level_mm.max():.1f} мм")