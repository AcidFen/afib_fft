# Detecção de Fibrilação Atrial por Intervalos RR (Tempo + FFT)

Projeto da disciplina de Sinais e Sistemas Lineares. O código detecta episódios de **fibrilação atrial (FA)** em ECG usando apenas os intervalos entre batimentos (RR), combinando duas condições:

1. **Domínio do tempo:** dispersão dos intervalos RR (coeficiente de variação).
2. **Domínio da frequência:** potência de alta frequência (0,15 a 0,5 Hz) do tacograma RR, calculada por **FFT**.

Os limiares de cada condição são calibrados com a média e o desvio padrão de janelas de ritmo normal e de FA. Um trecho só é classificado como FA quando **as duas condições são atendidas ao mesmo tempo** (E lógico).

Os dados vêm do [MIT-BIH Atrial Fibrillation Database](https://physionet.org/content/afdb/1.0.0/) (PhysioNet), baixado automaticamente pela biblioteca `wfdb`.

> Artigo: **[PREENCHER: link ou referência do artigo]**

## Como funciona

1. Os instantes dos picos R vêm das anotações da base (`.qrs`). Não há detecção de QRS no sinal.
2. Para cada instante T, usa-se a janela causal dos **30 s anteriores** (T − 30, T]. Intervalos RR fora de 0,2 a 3,0 s são descartados, e é preciso ter pelo menos 9 intervalos válidos.
3. Duas características por janela:
   - `f_t = ln(CV + 0,005)`, com `CV = desvio padrão do RR / média do RR`.
   - `f_f = ln(P_HF + 1e-6)`: o tacograma normalizado é reamostrado a 4 Hz, tem a tendência linear removida, recebe janela de Hann e passa pela FFT. `P_HF` é a potência de 0,15 a 0,5 Hz.
4. **Calibração:** com janelas 100% normais e 100% FA dos registros de treino (passo de 10 s), calcula-se a média e o desvio padrão de cada classe. O limiar é `L = (μ_N·σ_A + μ_A·σ_N) / (σ_N + σ_A)`.
5. **Decisão:** FA se `f_t > L_t` **e** `f_f > L_f`.
6. **Índice de FA (0 a 1):** sigmoide da menor das duas margens normalizadas. Passa de 0,5 apenas quando as duas condições são atendidas. É mostrado na tela como P(FA), mas **não é uma probabilidade calibrada**.

## Estrutura

```
.
├── afib_fft.py          # animação em tempo real + calibração + avaliação
├── gerar_figuras.py     # gera as figuras do artigo e imprime as Tabelas I e II
├── calibracao_fa.npz    # (gerado) cache dos limiares calibrados
├── figuras/             # (gerada) PNGs produzidos por gerar_figuras.py
└── README.md
```

## Requisitos

- Python 3.9 ou superior
- Internet (o AFDB é baixado direto do PhysioNet)

```bash
pip install numpy matplotlib wfdb
```

## Como usar

### 1) Animação em tempo real

```bash
python afib_fft.py
```

Na primeira execução o script calibra os limiares (baixa só as anotações, é leve) e salva `calibracao_fa.npz`. Nas seguintes, carrega o arquivo. Se algum parâmetro de análise mudar (janela, banda HF, taxa de reamostragem ou registros de treino), ele recalibra sozinho. Para **forçar a recalibração** e ver a tabela de médias, desvios e limiares no terminal, apague `calibracao_fa.npz`.

O terminal mostra também a avaliação no registro exibido (sensibilidade, especificidade e acurácia).

**O que aparece na janela:**

| Elemento | Significado |
|---|---|
| Linha cinza | ECG rolando, com 0 = agora |
| Triângulos vermelhos | Picos QRS |
| Linha verde rente ao eixo | Trechos de FA anotados por especialistas (referência) |
| Curva azul | Índice de FA estimado, usando os 30 s anteriores |
| Linha pontilhada em 0,5 | Limiar de decisão |
| Texto no topo | Ritmo anotado, bpm, P(FA), estado das duas condições (tempo e frequência) e tempo em minutos |

### 2) Figuras e tabelas do artigo

```bash
python gerar_figuras.py
```

Imprime no terminal:

- **Tabela I:** média ± desvio e limiar de cada característica, nas duas classes.
- **Tabela II:** VP, FN, VN, FP, sensibilidade, especificidade e acurácia no registro de teste.

E salva em `figuras/`:

| Arquivo | Conteúdo |
|---|---|
| `fig1_ecg_normal_vs_fa.png` | ECG com picos R, ritmo normal × FA |
| `fig2_tacograma_rr.png` | Intervalos RR, normal × FA |
| `fig3_psd_fft.png` | PSD (FFT) do tacograma, com a banda HF destacada |
| `fig4_media_variancia_janela.png` | Média e variância do RR em janela deslizante |
| `fig5_espectrograma_stft.png` | Espectrograma do tacograma ao redor do início da FA |
| `fig6_dispersao_limiares.png` | Janelas de calibração no plano tempo × frequência, com os limiares |

## Parâmetros principais

Ficam no topo de cada script. Se alterar um deles, altere nos dois para os resultados continuarem iguais.

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `REGISTRO` | `'04048'` | Registro do AFDB exibido e usado como teste (nunca entra na calibração) |
| `TREINO` | 7 registros | Registros usados para calibrar os limiares |
| `W_FEATS` / `W` | `30.0` s | Janela retrospectiva de análise |
| `MIN_RR` | `9` | Mínimo de intervalos RR válidos na janela |
| `FS_INTERP` / `FS_I` | `4.0` Hz | Taxa de reamostragem do tacograma antes da FFT |
| `F_HF` | `(0.15, 0.5)` Hz | Banda de alta frequência |
| `DURACAO`, `JANELA`, `FPS` | `60`, `6`, `25` | Reprodução: segundos totais, segundos visíveis e quadros por segundo |
| `INICIO` | `None` | Início da reprodução em segundos (`None` = 15 s antes da 1ª FA) |
| `CURVA_PASSO` | `0.25` s | Resolução da curva azul |

Para testar outro registro, mude `REGISTRO` e confirme que ele não está em `TREINO`.

## Uso offline (opcional)

Baixe o banco uma vez:

```python
import wfdb
wfdb.dl_database('afdb', 'afdb_local')
```

Depois troque `pn_dir='afdb'` por `dir_name='afdb_local'` nas chamadas `rdheader`, `rdann` e `rdrecord` dos dois scripts.

## Limitações

- Avaliação em **um único registro de teste**. Uma validação leave-one-record-out com todos os registros seria mais robusta.
- A janela de 30 s e o janelamento de Hann fazem o índice **demorar a subir** no início de um episódio de FA. A regra do E lógico acompanha a condição mais lenta.
- Janelas mistas (com transição de ritmo) ficam fora da calibração e da avaliação.
- Os picos R vêm das anotações da base. Em um sistema real, erros de detecção de QRS afetariam a série RR.
- A banda HF e a janela foram escolhidas de forma heurística, sem otimização.

## Próximos passos

- Janelas com menor atenuação nas pontas (por exemplo, Tukey) e janelas mais curtas.
- Comparação FFT × STFT × CWT.
- Classificadores de aprendizado de máquina (Random Forest, SVM) sobre as mesmas características.
- Detector de QRS a partir do sinal bruto (por exemplo, Pan-Tompkins).

## Dados e referências

- G. B. Moody e R. G. Mark, "A new method for detecting atrial fibrillation using R-R intervals", *Computers in Cardiology*, vol. 10, pp. 227–230, 1983. Base: [MIT-BIH Atrial Fibrillation Database](https://physionet.org/content/afdb/1.0.0/).
- A. L. Goldberger *et al.*, "PhysioBank, PhysioToolkit, and PhysioNet", *Circulation*, vol. 101, n. 23, pp. e215–e220, 2000.

## Autores

Grupo 4 - João Victor Lemes, Bryan, Felippe Teixeira, Victor Bueno, Felipe Maciel · Universidade Tecnológica Federal do Paraná
