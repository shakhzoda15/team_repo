import numpy as np
import matplotlib.pyplot as plt

L = 1.17

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

adcMeans = np.array(adcMeans)
levels = np.array(heights) / 1000
fit = np.polyfit(adcMeans, levels, 3)
minAdc = adcMeans.min()
maxAdc = adcMeans.max()


list_taus = {
    20: 1.56,
    40: 1.55,
    60: 1.15,
    80: 1.04,
    100: 1.20
}

def process_experiment_and_plot(filename, h_nominal):
    samples, duration, count = read(filename)
    level = np.polyval(fit, np.clip(samples, minAdc, maxAdc)) * 1000
    t = np.linspace(0, duration, count)

    tau = list_taus[h_nominal]

    mask_start = (t >= 0.2) & (t <= 0.5)
    mean_start = np.mean(level[mask_start])

    idx_start = np.argmin(np.abs(t - tau))
    idx_end = min(idx_start + int(0.3 * count / duration), count)
    fit_down = np.polyfit(t[idx_start:idx_end], level[idx_start:idx_end], 1)

    plt.figure(figsize=(8, 5))
    plt.plot(t, level, color='tab:blue', linewidth=0.8, alpha=0.8)

    plt.axhline(mean_start, color='k', linestyle=':')

    t_plot = np.linspace(max(0, tau - 0.5), t[idx_end], 50)
    plt.plot(t_plot, np.polyval(fit_down, t_plot), color='k', linewidth=1.5)

    plt.axvline(tau, color='r', linestyle='--')

    c_exp = L / tau
    c_theory = np.sqrt(9.81 * (h_nominal / 1000))
    text = (f"L = {L:.2f} [м]\n"
            f"τ = {tau:.2f} [с]\n"
            f"c = {c_exp:.2f} [м/с]\n"
            f"√(gh) = {c_theory:.2f} [м/с]")

    plt.text(0.95, 0.95, text,
             transform=plt.gca().transAxes,
             ha='right', va='top',
             bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})

    plt.title(f"wave {h_nominal} mm")
    plt.xlabel("время после открытия дверцы, с")
    plt.ylabel("уровень воды, мм")
    plt.grid(True, alpha=0.5)

    plt.xlim(0, 4.0)
    plt.xticks(np.arange(0, 4.5, 0.5))

    plt.tight_layout()
    plt.savefig(f"../plots/wave_{h_nominal}mm.png", dpi=300)
    plt.show()

    return tau


heights = [20, 40, 60, 80, 100]
speeds = []
taus = []

for h in heights:
    filename = dataPath + "wave%dmm.txt" % h
    tau = process_experiment_and_plot(filename, h)
    taus.append(tau)
    speeds.append(L / tau)

# теоретическая зависимость
speeds = np.array(speeds)
taus = np.array(taus)
h_meters = np.array(heights) / 1000.0
theory_speeds = np.sqrt(9.81 * h_meters)

plt.figure(figsize=(8, 5))
h_curve = np.linspace(0, 0.12, 100)
plt.plot(h_curve * 1000, np.sqrt(9.81 * h_curve), label="теория c = √(gh)")
plt.plot(heights, speeds, 'o', label="опыт")
plt.xlabel("глубина, мм")
plt.ylabel("скорость волны, м/с")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig("../plots/theory_curve.png", dpi=300)
plt.show()

print("\nтаблица")
print(f"{'Глубина, мм':<12} | {'τ, с':<6} | {'c, м/с':<8} | {'√(gh), м/с':<10} | {'Расхождение':<10}")
print("-" * 55)
for h, tau, c_exp, c_th in zip(heights, taus, speeds, theory_speeds):
    diff = (c_exp - c_th) / c_th * 100
    print(f"{h:<12} | {tau:<6.2f} | {c_exp:<8.2f} | {c_th:<10.2f} | {diff:+.1f}%")

# логарифмический график
valid = ~np.isnan(taus)
ln_h = np.log(h_meters[valid])
ln_c = np.log(speeds[valid])

coeffs, cov = np.polyfit(ln_h, ln_c, 1, cov=True)
k = coeffs[0]
b = coeffs[1]
err_k = np.sqrt(cov[0, 0])
err_b = np.sqrt(cov[1, 1])

print(f"\nНаклон: {k:.2f} ± {err_k:.2f} (теория 0.5)")
print(f"Свободный член: {b:.2f} ± {err_b:.2f} (теория 1.14)")

plt.figure(figsize=(8, 5))
plt.plot(ln_h, ln_c, 'o', label="опыт")
plt.plot(ln_h, k * ln_h + b, '-', label=f"наклон {k:.2f} ± {err_k:.2f}")
plt.xlabel("ln h")
plt.ylabel("ln c")
plt.title("Проверка теории в логарифмических координатах")
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.savefig("plots/log_log.png", dpi=300)
plt.show()
