"""
ECG em tempo real (MIT-BIH Atrial Fibrillation Database)
Detecção de fibrilação atrial por DUAS condições (E lógico):
  1) Domínio do TEMPO: dispersão (desvio padrão normalizado) dos intervalos RR
  2) Domínio da FREQUÊNCIA: potência de alta frequência do tacograma RR (via FFT)
Os limiares de cada condição são calibrados com trechos normais e trechos de FA
(média e desvio padrão de cada classe). Só é FA se as duas condições forem
atendidas ao mesmo tempo.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import wfdb

REGISTRO = '04048'
DURACAO = 60        # segundos do registro a reproduzir
JANELA = 6          # segundos visíveis na tela
FPS = 25            # quadros por segundo
INICIO = None       # None = começa 15 s antes da 1ª fibrilação; ou um valor em segundos

# ---------- Análise ----------
W_FEATS = 30.0      # janela (s) olhando para trás
MIN_RR = 9          # mínimo de intervalos RR válidos na janela
CURVA_PASSO = 0.25  # resolução da curva azul (s)
FS_INTERP = 4.0     # taxa (Hz) de reamostragem do tacograma RR antes da FFT
F_HF = (0.15, 0.5)  # banda de "alta frequência" do tacograma (Hz)
# Registros usados para calibrar (o REGISTRO exibido nunca entra)
TREINO = ['04043', '04048', '04126', '04746', '04908', '04936', '05091', '05121']
ARQ_MODELO = Path(__file__).with_name('calibracao_fa.npz')
NOMES = ['RR irregular (tempo)', 'HF alto (frequência)']


# =====================================================================
# Anotações
# =====================================================================
def trechos_afib(atr, fs, dur_total):
    """Ritmos anotados [(t, rótulo)] e trechos de FA [(início, fim)] em segundos."""
    ritmos = [(s / fs, n.strip('\x00 '))
              for s, n in zip(atr.sample, atr.aux_note) if n.strip('\x00 ')]
    trechos = []
    for i, (ini, nota) in enumerate(ritmos):
        fim = ritmos[i + 1][0] if i + 1 < len(ritmos) else dur_total
        if nota == '(AFIB':
            trechos.append((ini, fim))
    return ritmos, trechos


def fracao_afib(trechos, T, W=W_FEATS):
    """Fração da janela (T-W, T] anotada como FA."""
    ov = sum(max(0.0, min(fim, T) - max(ini, T - W)) for ini, fim in trechos)
    return ov / W


# =====================================================================
# Características: uma no tempo, outra na  (FFT)
# =====================================================================
def features_janela(qrs_t, T, W=W_FEATS):
    """Retorna [log(desvio RR normalizado), log(potência HF normalizada)]
    usando só os W segundos anteriores a T (causal)."""
    i1 = np.searchsorted(qrs_t, T, side='right')
    i0 = np.searchsorted(qrs_t, T - W, side='left')
    tb = qrs_t[i0:i1]
    rr = np.diff(tb)
    tt = tb[1:]                                       # instante de cada intervalo RR
    ok = (rr > 0.2) & (rr < 3.0)                      # descarta RR fisiologicamente absurdos
    rr, tt = rr[ok], tt[ok]
    if len(rr) < MIN_RR:
        return None
    m = rr.mean()

    # --- Domínio do tempo: desvio padrão dos RR normalizado pela média ---
    f_tempo = np.log(rr.std() / m + 0.005)

    # --- Domínio da frequência: FFT do tacograma RR reamostrado ---
    grade = np.arange(tt[0], tt[-1], 1.0 / FS_INTERP)
    if len(grade) < 32:
        return None
    r = (np.interp(grade, tt, rr) - m) / m            # tacograma normalizado, média zero
    r = r - np.polyval(np.polyfit(grade, r, 1), grade)  # remove tendência linear
    jan = np.hanning(len(r))
    X = np.fft.rfft(r * jan)
    psd = np.abs(X) ** 2 / (FS_INTERP * np.sum(jan ** 2))   # densidade espectral (1/Hz)
    f = np.fft.rfftfreq(len(r), 1.0 / FS_INTERP)
    banda = (f >= F_HF[0]) & (f < F_HF[1])
    p_hf = psd[banda].sum() * (f[1] - f[0])           # potência na banda de alta frequência
    f_freq = np.log(p_hf + 1e-6)

    return np.array([f_tempo, f_freq])


# =====================================================================
# Calibração dos limiares (normal x FA) e decisão
# =====================================================================
def calibrar(X, y):
    """Média e desvio padrão de cada característica nas duas classes.
    Limiar = ponto entre as médias ponderado pelos desvios (fica a mesma
    distância, em desvios padrão, das duas classes)."""
    N, A = X[y == 0], X[y == 1]
    mu_n, sd_n = N.mean(0), N.std(0) + 1e-9
    mu_a, sd_a = A.mean(0), A.std(0) + 1e-9
    lim = (mu_n * sd_a + mu_a * sd_n) / (sd_n + sd_a)
    return dict(mu_n=mu_n, sd_n=sd_n, mu_a=mu_a, sd_a=sd_a, lim=lim)


def condicoes(x, cal):
    """Cada condição é atendida quando a característica passa do limiar."""
    return x > cal['lim']


def indice_fa(x, cal):
    """Índice contínuo em [0,1]. Usa a PIOR das duas margens (E lógico):
    só passa de 0,5 se as duas características estão acima dos limiares."""
    margem = (x - cal['lim']) / ((cal['sd_n'] + cal['sd_a']) / 2)
    return 1.0 / (1.0 + np.exp(-1.5 * margem.min()))


def montar_dataset(rec, passo=10.0):
    """Janelas 'puras' (100% FA ou 0% FA) de um registro do PhysioNet."""
    hdr = wfdb.rdheader(rec, pn_dir='afdb')
    fs, dur = hdr.fs, hdr.sig_len / hdr.fs
    qrs_t = wfdb.rdann(rec, 'qrs', pn_dir='afdb').sample / fs
    _, trechos = trechos_afib(wfdb.rdann(rec, 'atr', pn_dir='afdb'), fs, dur)
    X, y = [], []
    for T in np.arange(W_FEATS, dur, passo):
        f = fracao_afib(trechos, T)
        if f > 0.999:
            rotulo = 1.0
        elif f < 0.001:
            rotulo = 0.0
        else:
            continue                                  # janela mista: ignorada
        x = features_janela(qrs_t, T)
        if x is not None:
            X.append(x)
            y.append(rotulo)
    return np.array(X), np.array(y)


def obter_calibracao(treino):
    chave = f'{W_FEATS}|{FS_INTERP}|{F_HF}|{",".join(treino)}'
    if ARQ_MODELO.exists():
        d = np.load(ARQ_MODELO)
        if str(d['chave']) == chave:
            print(f'Calibração carregada de {ARQ_MODELO.name}')
            return {k: d[k] for k in ('mu_n', 'sd_n', 'mu_a', 'sd_a', 'lim')}

    print('Calibrando limiares (baixa só as anotações, é leve)...')
    Xs, ys = [], []
    for rec in treino:
        try:
            X, y = montar_dataset(rec)
        except Exception as e:
            print(f'  {rec}: pulado ({e})')
            continue
        if len(y) == 0:
            continue
        print(f'  {rec}: {len(y)} janelas ({int(y.sum())} de FA)')
        Xs.append(X)
        ys.append(y)
    if not Xs:
        raise RuntimeError('Não foi possível montar o conjunto de calibração (sem internet?).')
    X, y = np.vstack(Xs), np.concatenate(ys)
    cal = calibrar(X, y)
    for j, nome in enumerate(NOMES):
        print(f'  {nome}: normal {cal["mu_n"][j]:.2f}±{cal["sd_n"][j]:.2f} | '
              f'FA {cal["mu_a"][j]:.2f}±{cal["sd_a"][j]:.2f} | limiar {cal["lim"][j]:.2f}')
    np.savez(ARQ_MODELO, chave=chave, **cal)
    return cal


def avaliar(qrs_t, trechos, dur, cal, passo=10.0):
    """Sensibilidade/especificidade nas janelas puras do registro exibido."""
    tp = fn = tn = fp = 0
    for T in np.arange(W_FEATS, dur, passo):
        f = fracao_afib(trechos, T)
        if 0.001 <= f <= 0.999:
            continue
        x = features_janela(qrs_t, T)
        if x is None:
            continue
        pred, real = bool(condicoes(x, cal).all()), f > 0.999
        tp += pred and real
        fn += (not pred) and real
        tn += (not pred) and (not real)
        fp += pred and (not real)
    sens = tp / (tp + fn) if tp + fn else float('nan')
    espec = tn / (tn + fp) if tn + fp else float('nan')
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    print(f'Avaliação no registro {REGISTRO} (fora da calibração): '
          f'sensibilidade {sens:.0%} | especificidade {espec:.0%} | acurácia {acc:.0%} '
          f'({tp + fn} janelas de FA, {tn + fp} sem FA)')


# =====================================================================
# Dados do registro exibido
# =====================================================================
hdr = wfdb.rdheader(REGISTRO, pn_dir='afdb')
fs = hdr.fs
dur_total = hdr.sig_len / fs
atr = wfdb.rdann(REGISTRO, 'atr', pn_dir='afdb')
qrs = wfdb.rdann(REGISTRO, 'qrs', pn_dir='afdb')
qrs_t = qrs.sample / fs

ritmos, trechos_fa = trechos_afib(atr, fs, dur_total)

print(f'{len(trechos_fa)} trecho(s) de AFIB no registro. Primeiros (s):')
for ini, fim in trechos_fa[:5]:
    print(f'  {ini:9.1f} -> {fim:9.1f}')

cal = obter_calibracao([r for r in TREINO if r != REGISTRO])
avaliar(qrs_t, trechos_fa, dur_total, cal)

if INICIO is None:
    INICIO = max(0, trechos_fa[0][0] - 15) if trechos_fa else 0
print(f'Reprodução começa em {INICIO:.1f} s')

sampfrom = int(INICIO * fs)
sampto = min(sampfrom + int(DURACAO * fs), hdr.sig_len)
rec = wfdb.rdrecord(REGISTRO, sampfrom=sampfrom, sampto=sampto, pn_dir='afdb')
sig = rec.p_signal[:, 0]
t = (sampfrom + np.arange(len(sig))) / fs

# Curva (causal): índice de FA e as duas condições a cada CURVA_PASSO segundos
tg = np.arange(t[0], t[-1], CURVA_PASSO)
pg = np.full(len(tg), np.nan)
cond_g = np.full((len(tg), 2), np.nan)
for k, T in enumerate(tg):
    x = features_janela(qrs_t, T)
    if x is not None:
        pg[k] = indice_fa(x, cal)
        cond_g[k] = condicoes(x, cal)

# =====================================================================
# Figura
# =====================================================================
passo = int(fs / FPS)
margem = 0.1 * (sig.max() - sig.min())

fig, ax = plt.subplots(figsize=(14, 5.5))
linha, = ax.plot([], [], color='#333333', lw=1.0, label='ECG')
pontos, = ax.plot([], [], 'rv', ms=8, ls='', label='QRS')

Y_FA = 0.02
fa_linha, = ax.plot([], [], color='green', lw=6, solid_capstyle='butt',
                    transform=ax.get_xaxis_transform(),
                    label='Fibrilação atrial anotada (AFIB)')

texto = ax.text(0.01, 0.95, '', transform=ax.transAxes, va='top',
                color='green', fontsize=12, fontweight='bold')

ax.set_xlim(-JANELA, 0)
ax.set_ylim(sig.min() - 2 * margem, sig.max() + margem)
ax.set_xlabel('Tempo (s, 0 = agora)')
ax.set_ylabel('Amplitude (mV)')
ax.set_title(f'ECG em tempo real - AFDB registro {REGISTRO}')
ax.grid(True, linestyle=':', alpha=0.6)

ax2 = ax.twinx()
prob_linha, = ax2.plot([], [], color='royalblue', lw=2.5,
                       label='Índice de FA (tempo E frequência)')
ax2.set_ylim(-0.15, 1.05)
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_ylabel('Índice de FA (>0,5 = as duas condições)', color='royalblue')
ax2.tick_params(axis='y', colors='royalblue')
ax2.axhline(0.5, color='royalblue', ls=':', alpha=0.4)

ax2.legend(handles=[linha, pontos, fa_linha, prob_linha], loc='upper center',
           bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False)
fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.24)


def update(frame):
    i1 = min((frame + 1) * passo, len(sig))
    i0 = max(0, i1 - int(JANELA * fs))
    agora = t[i1 - 1]

    linha.set_data(t[i0:i1] - agora, sig[i0:i1])

    visiveis = qrs_t[(qrs_t >= t[i0]) & (qrs_t <= agora)]
    pontos.set_data(visiveis - agora, np.interp(visiveis, t, sig))

    xs, ys = [], []
    for ini, fim in trechos_fa:
        a, b = max(ini, t[i0]), min(fim, agora)
        if b > a:
            xs += [a - agora, b - agora, np.nan]
            ys += [Y_FA, Y_FA, np.nan]
    fa_linha.set_data(xs, ys)

    m = (tg >= t[i0]) & (tg <= agora)
    prob_linha.set_data(tg[m] - agora, pg[m])
    if m.any() and np.isfinite(pg[m][-1]):
        c = cond_g[m][-1]
        p_txt = (f'tempo {"SIM" if c[0] else "não"} | '
                 f'freq {"SIM" if c[1] else "não"} | índice {pg[m][-1]:.0%}')
    else:
        p_txt = '--'

    vigente = [n for ts, n in ritmos if ts <= agora]
    recentes = qrs_t[qrs_t <= agora][-5:]
    bpm = f' | {60 / np.mean(np.diff(recentes)):.0f} bpm' if len(recentes) >= 2 else ''
    texto.set_text(f'Ritmo anotado: {vigente[-1] if vigente else "?"}{bpm}'
                   f'\nRR irregular / HF alto: {p_txt} | {agora / 60:.1f} min')
    return linha, pontos, fa_linha, prob_linha, texto


ani = FuncAnimation(fig, update, frames=len(sig) // passo,
                    interval=1000 / FPS, blit=True, repeat=False)
plt.show()
