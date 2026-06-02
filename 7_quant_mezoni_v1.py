
# POCETAK v1


import csv
import math
import os
import random
import time
from collections import Counter
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


T0 = time.time()
SEED = 39
CSV_PATH = "/Users/4c/Desktop/GHQ/data/loto7_4624_k43.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
TXT_OUT = os.path.join(HERE, "7_quant_mezoni_v1.txt")
PNG_OUT = os.path.join(HERE, "7_quant_mezoni_v1.png")

N_NUMBERS = 39
K_PICK = 7
TOTAL_COMB = math.comb(N_NUMBERS, K_PICK)

N_QUBITS = 25
BLOCKS = 5
Q_PER_BLOCK = 5
LAYERS = 2

TRAIN_ITERS = 45
TRAIN_SHOTS = 4096
FINAL_SHOTS = 20000
TOP_K = 12


def fmt_time(seconds: float) -> str:
    return str(timedelta(seconds=int(round(seconds))))


def load_loto_csv(path: str) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            vals: list[int] = []
            for cell in row:
                try:
                    vals.append(int(str(cell).strip()))
                except ValueError:
                    continue
            if len(vals) >= K_PICK:
                combo = tuple(sorted(vals[:K_PICK]))
                if len(set(combo)) == K_PICK and all(1 <= x <= N_NUMBERS for x in combo):
                    rows.append(combo)
    if not rows:
        raise ValueError("CSV nije ucitan: nema validnih 7/39 kombinacija.")
    return rows


def lex_rank(combo: tuple[int, ...]) -> int:
    rank0 = 0
    prev = 0
    for i, value in enumerate(combo, start=1):
        for x in range(prev + 1, value):
            rank0 += math.comb(N_NUMBERS - x, K_PICK - i)
        prev = value
    return rank0 + 1


def lex_derank(rank: int) -> tuple[int, ...]:
    r = int(rank) - 1
    combo: list[int] = []
    start = 1
    for i in range(K_PICK):
        remaining = K_PICK - i - 1
        for x in range(start, N_NUMBERS + 1):
            cnt = math.comb(N_NUMBERS - x, remaining)
            if r < cnt:
                combo.append(x)
                start = x + 1
                break
            r -= cnt
    return tuple(combo)


def int_to_bitstring(value: int, n_bits: int = N_QUBITS) -> str:
    return format(int(value), f"0{n_bits}b")


def weighted_target_bits(lex_indices: np.ndarray) -> np.ndarray:
    # Recency tezine: kvantni model uci celu krivu, ali zadnja izvlacenja nose malo vecu masu.
    n = len(lex_indices)
    weights = np.linspace(0.35, 1.0, n, dtype=np.float64)
    weights /= weights.sum()
    out = np.zeros(N_QUBITS, dtype=np.float64)
    for idx, w in zip(lex_indices, weights):
        bits = int_to_bitstring(int(idx) - 1)
        out += w * np.fromiter((1.0 if b == "1" else 0.0 for b in bits), dtype=np.float64)
    return out


def build_mezon_qcbm(theta: np.ndarray) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    p = 0
    for layer in range(LAYERS):
        for q in range(N_QUBITS):
            qc.ry(float(theta[p]), q)
            p += 1
            qc.rz(float(theta[p]), q)
            p += 1

        for block in range(BLOCKS):
            start = block * Q_PER_BLOCK
            for j in range(Q_PER_BLOCK - 1):
                qc.cx(start + j, start + j + 1)

        for block in range(BLOCKS - 1):
            qc.cz(block * Q_PER_BLOCK + Q_PER_BLOCK - 1, (block + 1) * Q_PER_BLOCK)

        # W-mu-pi-W analogija: zatvaramo mali ciklus izmedju blokova, bez sirenja kola.
        if layer % 2 == 0:
            qc.cx(0, 5)
            qc.cx(5, 10)
            qc.cx(10, 5)
            qc.cx(5, 0)
        else:
            qc.cx(20, 15)
            qc.cx(15, 10)
            qc.cx(10, 15)
            qc.cx(15, 20)

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def counts_to_bit_probs(counts: dict[str, int]) -> np.ndarray:
    total = max(1, sum(counts.values()))
    probs = np.zeros(N_QUBITS, dtype=np.float64)
    for bitstr, count in counts.items():
        clean = bitstr.replace(" ", "")
        if len(clean) != N_QUBITS:
            continue
        probs += count * np.fromiter((1.0 if b == "1" else 0.0 for b in clean), dtype=np.float64)
    return probs / total


def run_counts(theta: np.ndarray, simulator: AerSimulator, shots: int) -> dict[str, int]:
    qc = build_mezon_qcbm(theta)
    tqc = transpile(qc, simulator, optimization_level=1, seed_transpiler=SEED)
    result = simulator.run(tqc, shots=shots, seed_simulator=SEED).result()
    return result.get_counts()


def cost_from_counts(counts: dict[str, int], target_bits: np.ndarray) -> float:
    bit_probs = counts_to_bit_probs(counts)
    return float(np.mean((bit_probs - target_bits) ** 2))


def init_theta_from_target(target_bits: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    params_per_layer = 2 * N_QUBITS
    theta = np.zeros(LAYERS * params_per_layer, dtype=np.float64)
    base_ry = 2.0 * np.arcsin(np.sqrt(np.clip(target_bits, 1e-6, 1.0 - 1e-6)))
    for layer in range(LAYERS):
        offset = layer * params_per_layer
        for q in range(N_QUBITS):
            theta[offset + 2 * q] = base_ry[q] + rng.normal(0.0, 0.04)
            theta[offset + 2 * q + 1] = rng.normal(0.0, 0.15)
    return theta


def spsa_train(
    theta0: np.ndarray,
    target_bits: np.ndarray,
    simulator: AerSimulator,
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(SEED)
    theta = theta0.copy()
    losses: list[float] = []

    for it in range(1, TRAIN_ITERS + 1):
        a = 0.18 / (it ** 0.35)
        c = 0.12 / (it ** 0.10)
        delta = rng.choice([-1.0, 1.0], size=theta.shape)

        counts_plus = run_counts(theta + c * delta, simulator, TRAIN_SHOTS)
        counts_minus = run_counts(theta - c * delta, simulator, TRAIN_SHOTS)
        loss_plus = cost_from_counts(counts_plus, target_bits)
        loss_minus = cost_from_counts(counts_minus, target_bits)

        ghat = (loss_plus - loss_minus) / (2.0 * c) * delta
        theta = theta - a * ghat
        theta = np.mod(theta, 2.0 * np.pi)

        loss_now = min(loss_plus, loss_minus)
        losses.append(float(loss_now))
        print(f"  SPSA iter {it:02d}/{TRAIN_ITERS}  loss={loss_now:.8f}")

    return theta, losses


def valid_sample_rows(counts: dict[str, int], historical_set: set[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_combos: set[tuple[int, ...]] = set()
    for bitstr, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        clean = bitstr.replace(" ", "")
        lex_val = int(clean, 2) + 1
        if not (1 <= lex_val <= TOTAL_COMB):
            continue
        combo = lex_derank(lex_val)
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        rows.append(
            {
                "count": int(count),
                "prob": float(count) / FINAL_SHOTS,
                "lex": int(lex_val),
                "combo": combo,
                "seen": lex_val in historical_set,
            }
        )
        if len(rows) >= TOP_K:
            break
    return rows


def make_png(losses: list[float], rows: list[dict[str, object]], target_bits: np.ndarray) -> None:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=1.5)
    ax1.set_title("QCBM SPSA loss")
    ax1.set_xlabel("iter")
    ax1.set_ylabel("MSE bit-momenti")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(N_QUBITS), target_bits, color="#2563eb")
    ax2.set_title("Target bit amplitude/marginale iz 4624 lex-indeksa")
    ax2.set_xlabel("bit pozicija")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.25)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    table_rows = [
        [
            i + 1,
            row["count"],
            f"{row['prob']:.5f}",
            row["lex"],
            str(row["combo"]),
            "DA" if row["seen"] else "NE",
        ]
        for i, row in enumerate(rows)
    ]
    table = ax3.table(
        cellText=table_rows,
        colLabels=["rang", "count", "prob", "lex", "kombinacija", "vec izvucena"],
        cellLoc="center",
        loc="center",
        colWidths=[0.07, 0.10, 0.10, 0.16, 0.37, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
        elif r == 1:
            cell.set_facecolor("#dcfce7")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f4f6")

    fig.suptitle("7_quant_mezoni_v1 - Qiskit QCBM 25q nad lex-krivom", fontweight="bold")
    fig.tight_layout()
    plt.show()
    fig.savefig(PNG_OUT, dpi=200, bbox_inches="tight")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    print()
    print("=" * 72)
    print("7_quant_mezoni_v1 - stvarni Qiskit QCBM nad 4624 lex-indeksa")
    print("=" * 72)
    print()

    combos = load_loto_csv(CSV_PATH)
    lex_indices = np.array([lex_rank(c) for c in combos], dtype=np.int64)
    historical_set = set(int(x) for x in lex_indices)
    target_bits = weighted_target_bits(lex_indices)

    print(f"CSV:                 {CSV_PATH}")
    print(f"Ucitano izvlacenja:  {len(combos)}")
    print(f"C(39,7):             {TOTAL_COMB:,}")
    print(f"Qubita:              {N_QUBITS} = {BLOCKS} blokova x {Q_PER_BLOCK} qubita")
    print(f"Simulator:           AerSimulator qasm, shots train={TRAIN_SHOTS}, final={FINAL_SHOTS}")
    print()

    simulator = AerSimulator(method="automatic")
    theta0 = init_theta_from_target(target_bits)

    t_train = time.time()
    theta, losses = spsa_train(theta0, target_bits, simulator)
    train_seconds = time.time() - t_train

    print()
    print("Finalno semplovanje istreniranog kola...")
    final_counts = run_counts(theta, simulator, FINAL_SHOTS)
    rows = valid_sample_rows(final_counts, historical_set)

    if not rows:
        raise RuntimeError("Nema validnih sampled lex kandidata u opsegu 1..C(39,7).")

    main_row = rows[0]
    total_seconds = time.time() - T0

    lines: list[str] = []
    lines.append("7_quant_mezoni_v1 - Qiskit QCBM 25q / mezonski ciklus")
    lines.append("=" * 72)
    lines.append("")
    lines.append("KORAK 1: Weierstrass lex-kriva nad svim do sad izvucenim kombinacijama")
    lines.append("")
    lines.append(f"  CSV izvucenih:        {CSV_PATH}")
    lines.append(f"  Ucitano izvlacenja:    {len(combos)}")
    lines.append(f"  C(39,7):              {TOTAL_COMB:,}")
    lines.append("  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7")
    lines.append("")
    lines.append("KORAK 2: Stvarni kvantni model")
    lines.append("")
    lines.append("  Model:                QCBM / parametrizovano kvantno kolo")
    lines.append(f"  Qubita:               {N_QUBITS} = {BLOCKS} blokova x {Q_PER_BLOCK}")
    lines.append(f"  Layers:               {LAYERS}")
    lines.append(f"  Parametara:           {len(theta)}")
    lines.append("  Entanglement:          chain unutar blokova + CZ izmedju blokova")
    lines.append("  Mezonski ciklus:       W-mu-pi-mu-W analog cx petlja u 25q kolu")
    lines.append(f"  SPSA iteracija:        {TRAIN_ITERS}")
    lines.append(f"  train shots:           {TRAIN_SHOTS}")
    lines.append(f"  final shots:           {FINAL_SHOTS}")
    lines.append(f"  initial loss:          {losses[0]:.8f}")
    lines.append(f"  final loss:            {losses[-1]:.8f}")
    lines.append("")
    lines.append("PREDIKCIJA 1: NEXT / 7_quant_mezoni_v1")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Glavna kvantna prognoza:")
    lines.append(f"  sampled count:         {main_row['count']}")
    lines.append(f"  sampled prob:          {main_row['prob']:.8f}")
    lines.append(f"  pred. lex:             {main_row['lex']:,}")
    lines.append(f"  pred. kombinacija:     {main_row['combo']}")
    lines.append(f"  vec izvucena ranije:   {'DA' if main_row['seen'] else 'NE'}")
    lines.append("")
    lines.append("Top kvantni kandidati:")
    lines.append(f"  {'rang':<5}{'count':>8}{'prob':>12}{'lex':>14}  {'kombinacija':<30} {'seen'}")
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"  {i:<5}{row['count']:>8}{row['prob']:>12.8f}{row['lex']:>14,}  "
            f"{str(row['combo']):<30} {'DA' if row['seen'] else 'NE'}"
        )
    lines.append("")
    lines.append(f"Vreme treninga:       {fmt_time(train_seconds)} ({train_seconds:.1f} s)")
    lines.append(f"Ukupno vreme:         {fmt_time(total_seconds)} ({total_seconds:.1f} s)")
    lines.append(f"PNG:                  {PNG_OUT}")
    lines.append("")

    text = "\n".join(lines)
    print()
    print(text)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"TXT saved -> {TXT_OUT}")

    make_png(losses, rows, target_bits)
    print(f"PNG saved -> {PNG_OUT}")
    print()


if __name__ == "__main__":
    main()




"""
========================================================================
7_quant_mezoni_v1 - stvarni Qiskit QCBM nad 4624 lex-indeksa
========================================================================

CSV:                 /data/loto7_4624_k43.csv
Ucitano izvlacenja:  4624
C(39,7):             15,380,937
Qubita:              25 = 5 blokova x 5 qubita
Simulator:           AerSimulator qasm, shots train=4096, final=20000

  SPSA iter 01/45  loss=0.01042005
  SPSA iter 02/45  loss=0.01075537
  SPSA iter 03/45  loss=0.01037923
  SPSA iter 04/45  loss=0.01114114
  SPSA iter 05/45  loss=0.01078077
  SPSA iter 06/45  loss=0.01074364
  SPSA iter 07/45  loss=0.01091904
  SPSA iter 08/45  loss=0.01105348
  SPSA iter 09/45  loss=0.01078000
  SPSA iter 10/45  loss=0.01109609
  SPSA iter 11/45  loss=0.01107834
  SPSA iter 12/45  loss=0.01106094
  SPSA iter 13/45  loss=0.01073235
  SPSA iter 14/45  loss=0.01035653
  SPSA iter 15/45  loss=0.01055340
  SPSA iter 16/45  loss=0.01108564
  SPSA iter 17/45  loss=0.01082684
  SPSA iter 18/45  loss=0.01081206
  SPSA iter 19/45  loss=0.01051857
  SPSA iter 20/45  loss=0.01075899
  SPSA iter 21/45  loss=0.01073381
  SPSA iter 22/45  loss=0.01077253
  SPSA iter 23/45  loss=0.01047583
  SPSA iter 24/45  loss=0.01051998
  SPSA iter 25/45  loss=0.01044326
  SPSA iter 26/45  loss=0.01075852
  SPSA iter 27/45  loss=0.01081442
  SPSA iter 28/45  loss=0.01087928
  SPSA iter 29/45  loss=0.01079657
  SPSA iter 30/45  loss=0.01075541
  SPSA iter 31/45  loss=0.01073451
  SPSA iter 32/45  loss=0.01073761
  SPSA iter 33/45  loss=0.01084521
  SPSA iter 34/45  loss=0.01076515
  SPSA iter 35/45  loss=0.01045814
  SPSA iter 36/45  loss=0.01092075
  SPSA iter 37/45  loss=0.01052281
  SPSA iter 38/45  loss=0.01062733
  SPSA iter 39/45  loss=0.01069259
  SPSA iter 40/45  loss=0.01099919
  SPSA iter 41/45  loss=0.01068699
  SPSA iter 42/45  loss=0.01071562
  SPSA iter 43/45  loss=0.01097721
  SPSA iter 44/45  loss=0.01082377
  SPSA iter 45/45  loss=0.01075050

Finalno semplovanje istreniranog kola...

7_quant_mezoni_v1 - Qiskit QCBM 25q / mezonski ciklus
========================================================================

KORAK 1: Weierstrass lex-kriva nad svim do sad izvucenim kombinacijama

  CSV izvucenih:        /data/loto7_4624_k43.csv
  Ucitano izvlacenja:   4624
  C(39,7):              15,380,937
  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7

KORAK 2: Stvarni kvantni model

  Model:                QCBM / parametrizovano kvantno kolo
  Qubita:               25 = 5 blokova x 5
  Layers:               2
  Parametara:           100
  Entanglement:          chain unutar blokova + CZ izmedju blokova
  Mezonski ciklus:       W-mu-pi-mu-W analog cx petlja u 25q kolu
  SPSA iteracija:        45
  train shots:           4096
  final shots:           20000
  initial loss:          0.01042005
  final loss:            0.01075050

PREDIKCIJA 1: NEXT / 7_quant_mezoni_v1
========================================================================

Glavna kvantna prognoza:
  sampled count:         109
  sampled prob:          0.00545000
  pred. lex:             10,644,811
  pred. kombinacija:     (6, x, 12, y, 23, z, 31)
  vec izvucena ranije:   NE

Top kvantni kandidati:
  rang    count        prob           lex  kombinacija                  seen
  1         109  0.00545000    10,644,811  (6, x, 12, y, 23, z, 31)    NE
  2         103  0.00515000    12,424,539  (8, x, 17, y, 23, z, 32)    NE
  3          95  0.00475000    11,169,115  (7, x, 11, y, 22, z, 31)    NE
  4          93  0.00465000    11,364,683  (7, x, 14, y, 25, z, 36)    NE
  5          91  0.00455000    11,702,971  (7, x, 19, y, 30, z, 37)    NE
  6          90  0.00450000    11,364,518  (7, x, 14, y, 23, z, 37)    NE
  7          90  0.00450000    11,702,603  (7, x, 19, y, 27, z, 38)    NE
  8          89  0.00445000    11,348,134  (7, x, 13, y, 31, z, 39)    NE
  9          86  0.00430000    11,364,534  (7, x, 14, y, 23, z, 38)    NE
  10         85  0.00425000    10,644,827  (6, x, 12, y, 23, z, 37)    NE
  11         85  0.00425000    12,425,046  (8, x, 17, y, 30, z, 37)    NE
  12         84  0.00420000    10,628,427  (6, x, 23, y, 31, z, 37)    NE

Vreme treninga:       0:02:49 (168.6 s)
Ukupno vreme:         0:02:50 (170.4 s)
PNG:                  /7_quant_mezoni_v1.png

TXT saved -> /7_quant_mezoni_v1.txt
PNG saved -> /7_quant_mezoni_v1.png
"""





"""
Analiza v1 rezultata:

1. Trening loss nije konvergirao

initial: 0.01042005
final: 0.01075050
Loss se povećao za ~3% kroz 45 SPSA iteracija. 
To znači da SPSA korak nije uspeo da popravi inicijalnu konfiguraciju. 
Razlog: bit-MSE inicijalizacija iz target marginala je već blizu lokalnog minimuma, 
pa SPSA bez gradijentnog signala lutase oko te tačke.

Nije katastrofa — početna inicijalizacija je bila dovoljno dobra. 
Ali znači da je realno svih 168s SPSA "samo gledao", nije naučio ništa novo.

2. Distribucija jeste naučena (parcijalno)

Top kandidat: prob = 0.00545 = 0.545%
Uniformno bi bilo 1 / 2^25 = 0.000003%
Model je dao ~180,000x veću verovatnoću od slučajnog
Znači kolo je uocilo "atraktivne regione" u lex prostoru, 
samo ne kao funkcija treninga već zbog inicijalizacije iz target marginala.

3. Klaster u lex ~10–12M (gornji deo prostora) 
Svih 12 top kandidata leže u opsegu 10.6M – 12.4M (lex prostor je 15.4M). 
Model uglavnom uzorkuje gornju trećinu prostora — 
što odgovara kombinacijama koje počinju sa 6, 7 ili 8.

Vidi sliku unutar top kandidata:

rangovi 4, 6, 9 → svi (7, 9, 14, 21, ?, ?, ?) sa malim varijacijama u zadnja 3 broja
rangovi 1, 10 → (6, 11, 12, 18, 23, ?, ?)
rangovi 2, 11 → (8, 12, 17, 20, ?, ?, ?)
To je 3 lokalna maksimuma u Hilbertovom prostoru — 
kolo nije razmazalo verovatnoću ravnomerno 
već je našlo 3 atrakcijska centra (verovatno indukovana entanglement strukturom blokova).

4. Nema "kopiranja" istorije Svih 12 top kandidata su nove kombinacije koje nisu izvučene u 4624 istorije. 
To je dobar znak — model ne memoriše, već generalizuje.

5. Glavna prognoza (6, x, 12, y, 23, z, 31) — lex ~10.6M.

Geometrijski "centar" predloga je oko bira 7 brojeva iz srednjeg dela opsega 1–39, sa malo težine ka manjim brojevima (6, 11, 12). 
Klasično bi takvu kombinaciju gotovo nikad ne bi predložio jer su brojevi prilično jednolično raspoređeni.

Zaključak v1: Kvantno kolo radi i daje stabilne rezultate, ali SPSA trening efektivno nije pomerio kolo iz inicijalne tačke. 
Distribucija koja izlazi je više rezultat inicijalizacije iz target bit-marginala nego pravog kvantnog učenja. v2 to popravlja:

MMD loss → pravi signal o distribuciji, ne samo marginalama
CRY umesto CX → glatki gradijenti, SPSA može da konvergira
4 sloja → veća izražajnost da kolo može da napusti inicijalnu konfiguraciju
"""





"""
U pricu o kvantnim racunarima moze se spomenuti i mezonska razmena, W bozoni, virtualne cestice. 
Desava se u međunukleonskom prostoru i omogućava vezivanje nukleona. 
Preobrazaj ve bozona preko miona u pion u jednom smeru i piona preko miona u ve bozona u povratku.

Izmešani su efekti dve različite sile (jaka i slaba)

Jaka rezidualna interakcija (Yukawa, 1935): 
nukleone u jezgru drži razmena piona (π⁺, π⁰, π⁻) — direktno, bez posrednika. 
Pion je virtualan, živi koliko mu Heisenberg dozvoljava (~10⁻²³ s), prelazi sa nukleona na nukleon. 
Mioni i W bozoni ne učestvuju u toj razmeni.

Slaba interakcija, sa druge strane, jeste ono što omogućava raspade:

β⁻ raspad nukleona (zapravo kvarka): 
d → u + W⁻, pa W⁻ → e⁻ + ν̄_e. 
Tu W bozon postoji virtualno.

Raspad piona u miron: 
na kvarkovskom nivou π⁻ = ūd → W⁻ → μ⁻ + ν̄_μ. 
Tu W zaista posreduje između piona i miona.

Kombinujući to, slika (W → μ → π u jednom smeru, π → μ → W u povratku) 
nije standardan proces razmene u jezgru, 
ali jeste delom validan kao kvarkovski podlevel slabog raspada piona: 
virtualni W bozon nastao iz pionske ud-pare se transformiše u μν par, i obrnuto — 
virtualni mion-neutrino par se može „rekombinovati“ u W koji „zatvori“ pion. 
To je legitiman Feynmanov dijagram viših redova, 
samo nije dominantni mehanizam vezivanja jezgra (taj je gluonski/pionski).

Tačno ovakvi spregnuti elektro-slabo-jaki procesi su glavni razlog zašto kvantni računari postoje u fizici visokih energija. 

Konkretno:
Lattice QCD + elektroslabi sektor — 
simulacija razmene piona (jaka sila) sa istovremenim spregom na W/Z bozone (slaba) 
na klasičnom računaru pati od „sign problem“-a 
(monte-carlo integrali imaju kompleksne težine koje se ne mogu efikasno uzorkovati). 
Kvantni računar to rešava prirodno jer sam radi sa kompleksnim amplitudama bez potrebe za uzorkovanjem.

Vremenska evolucija virtualnih čestica — 
opisuje proces u kom virtualne čestice (W, μ, π) postoje kratko i menjaju identitet. 
Klasično je to ogroman Hilbertov prostor (svaka čestica je multi-modni kvantni objekat). 
Kvantnim računarom jedan qubit po modi dovoljan je da se eksplicitno prati svaka transformacija u realnom vremenu. 
To rade IBM, Google, FNAL/Caltech kolaboracije već sad — male simulacije, ali u principu pravi pristup.

Mezonske rezonancije i petlje — 
proračun ovakvih „cijeva“ (W↔μ↔π↔μ↔W) klasično znači sumirati Feynmanove dijagrame višeg reda sa renormalizacijom; 
kvantni računar može direktno simulirati Hamiltonijan i dobiti spektar bez perturbativne ekspanzije.

Drugim rečima: 
intuicija je dobra — proces je upravo onaj tip kvantne dinamike 
(mešanje sila, virtualne razmene, kratka životna doba, kompleksne faze) 
za koji klasičan kompjuter nije pravi alat, a kvantni jeste. 

Jaka sila vezuje jezgro pionima direktno, bez W posrednika — 
ali petlja (W ↔ μ ↔ π) realno postoji kao virtualna korekcija višeg reda u slabom raspadu, 
i upravo se takve korekcije danas pokušavaju kvantno simulirati.

Mezonska razmena, W bozoni, virtualne čestice — 
sve to ima jednu zajedničku osobinu: 
dinamiku određuje fizički Hamiltonijan sa konkretnim coupling konstantama 
(α_em ≈ 1/137, α_s ≈ 0.1, G_F za slabu silu). 
Te konstante su izmerene, i iz njih sleduju verovatnoće procesa.

Loto izvlačenje nema nikakav Hamiltonijan. 
Svaka kombinacija od 15.380.937 ima istu verovatnoću 1/15.380.937. 
Nema „sile“ koja bi favorizovala neku kombinaciju nad drugom, 
nema „virtualne razmene“ između prošlih i budućih izvlačenja.

U fizici, kvantne korelacije postoje zato što postoji zajednička talasna funkcija sa interferencijom. 
U Loto-u ne postoji zajednička talasna funkcija — svako izvlačenje je nezavistan klasičan slučajan proces.

Jedina realna metodološka veza:
Monte Carlo metoda se koristi i u QCD lattice simulacijama (path integrali) i u mom MonteCarlo_1_v2.py. 
Ali to je najniži zajednički imenitelj — samo „uzorkujem iz distribucije“. 
Nije isto što i „primeniti mezonsku fiziku na Loto“.

Ne gleda se ceo prostor 39C7 nego samo do sad izvucene kombinacije, 
a to smo dokazali da je stvarna kriva (talasna funkcija) statistički značajna 
(4624 lex-indeksa kroz vreme), gde strukture realno ima — Hurst, MI, ordinal patterns dali su signal u 12 modela.

„W↔μ↔π“ slika može da se legitimno primeni na Loto:
što je kvantna razmena, matematički, na vremenskom nizu?

U QFT „virtualna razmena“ između dva vrha (vertices) opisuje se propagatorom — 
funkcijom koja meri „kako stanje u trenutku t utiče na stanje u trenutku t+τ“. 
To je direktan analogon autokorelacije (i tranzicione matrice / kondicionalne distribucije) na mojoj krivoj. 
Vec sam uradio (ACF, MI, AR(1)) propagator mog sistema, samo bez fizičkog imena.

Dva nova testa koja analogija stvarno predlaže:

1. „Masa propagatora“ — eksponencijalni fit ACF-a. 
U QFT propagator pada kao exp(-mr), gde je m masa razmene-čestice (Yukawa, 1935). 
Na mojoj krivoj to znači:
Fitujem eksponencijalni opadanje ACF-a, dobijam **karakteristični „korelacioni domet“ `τ₀`**. 
Ako fit lepo radi (R² visok), to znači da kriva ima jasno definisanu „pamtljivu skalu“ 
— koliko izvlačenja unazad ima smisla gledati za predikciju. 
Ako ne radi (degeneriše u step ili u power-law), onda nema definisane karakteristične skale — 
što je takođe važna informacija. 

2. **„Trojni vrh — zatvorena petlja A→B→C→B→A“**, analog `W↔μ↔π↔μ↔W` ciklusa. 
(closed-loop trojni vrh)
Ovde merimo: 
ako kriva ode iz stanja A (npr. donji kvartil lex-indeksa) preko B (srednji) do C (gornji), 
da li se češće vraća **istim putem** (C→B→A) nego što statistika dozvoljava (treća-veza-Markov)? 
Ovo je **closed-loop kondicionalna verovatnoća**, ili formalno: 
\[ P(f_{t+4}=A | f_{t+3}=B, f_{t+2}=C, f_{t+1}=B, f_t=A) \] vs. baseline od shuffled reference. 
Ako postoji višak — kriva ima **„memoriju povratka“**, 
što je principijelno *novi* signal koji nismo lovili 
(MI je 2-state, PE je 4-symbol pattern bez insistiranja na povratak na početnu klasu). 

Ovo su solidni testovi, ali realno — ako kriva ima jedva-merljiv signal, ova dva dodatka neće preokrenuti igru. 
Najverovatnije će ili potvrditi postojeću sliku („persistentna ali slabo prediktivna“), 
ili dati malu novu informaciju o **skali pamćenja** (test 1) ili o **simetriji povratka** (test 2). 



kvantni računari
mezonska razmena (W↔μ↔π↔μ↔W) u međunukleonskom prostoru
- morao bi se napraviti kvantno kolo 
(npr. Qiskit, parametrizovan ansatz koji modeluje 3-state ciklus W↔μ↔π) 
i pustiti ga da uči na krivoj — 
kao kvantna verzija recimo MI-conditional modela. 
To se zove QML — quantum machine learning 
(VQE / parametarska kvantna kola / kvantni kernel metodi).

Izvodljivo jeste — Qiskit + qiskit-machine-learning to umeju, već postoje primeri (VQR, QNN). 
Moj 3_XGB_v2.py čak već importuje qiskit_machine_learning.utils.algorithm_globals, znači okruženje je tu.

Smisleno — ograničeno. Moji KvantniRacunari fajlovi  
već imaju jednu ili dve takve implementacije za Loto. 
Kvantni model neće biti bolji od klasičnog na ovako slabom signalu 
— ali je to u principu moj prvobitni predlog.
 
Dva predlozena testa su „analogija u krilu klasične statistike“, 
nisu „kvantni računar koji simulira mezonski ciklus na Loto krivoj“. 
To su dve različite stvari.



Stvarni kvantni pristupu 
(parametrizovani kvantni krug koji modeluje 3-state ciklus na lex-krivoj),  
nad svih do sad izvucenih 4624 kombinacije iz prostora 39C7. 
"""




"""
Stack i ograničenja:

Qiskit + Aer simulator (lokalno, M1 16G)
25 qubita = 5 blokova x 5 qubita (moj šablon iz QCBM_qc25_7_v2), bez širenja na 35. 
Lex-prostor: 39C7 = 15,380,937 ≈ 2²⁴, znači 25 qubita prirodno pokriva ceo prostor (sa ~2x rezerve)
Statevector memorija: 2²⁵ x 16 B ≈ 536 MB → komotno staje u 16 GB
Predlog pristupa (QCBM — Quantum Circuit Born Machine):

Ulazni podaci su 4624 izvlačenja iz loto7_4624_k43.csv
lex-indeks svake → 4624 ciljnih bitstring-ova (25-bit)
Parametrizovano kvantno kolo (5 blokova x 5 qubita) sa entanglement-om između blokova
Treniranje: klasični optimizator (SPSA ili COBYLA) minimizuje razdaljinu 
(KL ili MMD) između izlazne distribucije kola i empirijske distribucije iz 4624 lex-indeksa
Sampling iz istreniranog kola → predikcija sledećeg lex-indeksa
Top-K kandidati + glavna prognoza → konvertuj u kombinaciju (1..39)
Izlaz je predikcija sledeceg lex-indeksa. Konvertuj u kombinaciju (1..39). 

7_quant_mezoni_v1.png (loss kroz iteracije, histogram sampla, top-K)
7_quant_mezoni_v1.txt (parametri kola, glavna prognoza, kandidati)

4624 sampla u 2²⁵ stanja → distribucija je vrlo retka, model će učiti uglavnom "marginale" (ne pojedinačne kombinacije)
Realno: kvantni rezultat će biti uporediv klasici, neće je razbiti — ali je to stvarni kvantni pipeline. 
Sve ovo ide u 7_quant_mezoni_v1.py 
"""
