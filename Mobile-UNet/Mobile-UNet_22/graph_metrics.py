import glob
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Buscar archivo ─────────────────────────────────────────────────────────────
files = sorted(glob.glob("training_log_*.txt"))

if not files:
    print("Error: no se encontró ningún archivo 'training_log_*.txt' en el directorio actual.")
    sys.exit(1)

if len(files) == 1:
    filepath = files[0]
    print(f"Archivo encontrado: {filepath}")
else:
    print("Archivos encontrados:")
    for i, f in enumerate(files):
        print(f"  [{i}] {f}")
    idx = int(input("Selecciona el número del archivo a graficar: "))
    filepath = files[idx]

# ── Leer datos ─────────────────────────────────────────────────────────────────
df = pd.read_csv(filepath)
df.columns = df.columns.str.strip()

required = {"Epoch", "Loss", "IOU_Train", "IOU_Val"}
missing = required - set(df.columns)
if missing:
    print(f"Error: faltan columnas en el archivo: {missing}")
    print(f"Columnas encontradas: {list(df.columns)}")
    sys.exit(1)

epochs     = df["Epoch"]
loss       = df["Loss"]
iou_train  = df["IOU_Train"]
iou_val    = df["IOU_Val"]

# ── Estilo ─────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

C_LOSS      = "#E24B4A"
C_TRAIN     = "#378ADD"
C_VAL       = "#1D9E75"
C_ANNOT     = "#444441"
LW          = 2.2
MS          = 6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Métricas de entrenamiento — {filepath}", fontsize=13, fontweight="bold", y=1.02)

# ── Gráfica 1: Loss ────────────────────────────────────────────────────────────
ax1.plot(epochs, loss, color=C_LOSS, marker="o", markersize=MS,
         linewidth=LW, label="Loss")

min_e = loss.idxmin()
ax1.scatter(epochs[min_e], loss[min_e], color=C_LOSS, s=80, zorder=5)
ax1.annotate(f"mín {loss[min_e]:.4f}\nép. {int(epochs[min_e])}",
             xy=(epochs[min_e], loss[min_e]),
             xytext=(8, 10), textcoords="offset points",
             fontsize=8.5, color=C_LOSS,
             arrowprops=dict(arrowstyle="->", color=C_LOSS, lw=1.1))

ax1.set_title("Loss por época", fontsize=12)
ax1.set_xlabel("Época")
ax1.set_ylabel("Loss")
ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax1.legend(framealpha=0.85)

# ── Gráfica 2: IOU Train vs Val ────────────────────────────────────────────────
ax2.plot(epochs, iou_train, color=C_TRAIN, marker="o", markersize=MS,
         linewidth=LW, label="IOU Train")
ax2.plot(epochs, iou_val,   color=C_VAL,   marker="s", markersize=MS,
         linewidth=LW, linestyle="--", label="IOU Val")

# Máximos
max_train = iou_train.idxmax()
max_val   = iou_val.idxmax()

for idx_m, col, c in [(max_train, iou_train, C_TRAIN), (max_val, iou_val, C_VAL)]:
    ax2.scatter(epochs[idx_m], col[idx_m], color=c, s=80, zorder=5)
    ax2.annotate(f"máx {col[idx_m]:.4f}\nép. {int(epochs[idx_m])}",
                 xy=(epochs[idx_m], col[idx_m]),
                 xytext=(8, -18), textcoords="offset points",
                 fontsize=8.5, color=c,
                 arrowprops=dict(arrowstyle="->", color=c, lw=1.1))

ax2.set_title("IOU_train, IOU_val/epoca", fontsize=12)
ax2.set_xlabel("epoca")
ax2.set_ylabel("IOU")
ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax2.legend(framealpha=0.85)

# ── Resumen en consola ─────────────────────────────────────────────────────────
print(f"\n{'─'*40}")
print(f"  Épocas totales : {len(df)}")
print(f"  Loss mínima    : {loss.min():.4f}  (época {int(epochs[loss.idxmin()])})")
print(f"  IOU Train máx  : {iou_train.max():.4f}  (época {int(epochs[iou_train.idxmax()])})")
print(f"  IOU Val máx    : {iou_val.max():.4f}  (época {int(epochs[iou_val.idxmax()])})")
print(f"{'─'*40}\n")

# ── Guardar ────────────────────────────────────────────────────────────────────
out_name = filepath.replace(".txt", "_metrics.png")
plt.tight_layout()
plt.savefig(out_name, dpi=150, bbox_inches="tight")
plt.show()
print(f"Gráfica guardada como: {out_name}")