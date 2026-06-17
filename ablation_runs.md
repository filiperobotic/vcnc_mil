# Ablation Runs — VCNC gate × OA-MIL

Quatro runs combinando o gate VCNC (config FIXED) com o pipeline OA-MIL
completo, em 2 cenários de ruído × 2 seeds. Baselines (VCNC sozinho,
OA-MIL sozinho) já existentes; aqui só a combinação.

## 1. Pré-requisitos

Os configs já apontam para o dataset correto de cada cenário via a
cadeia de `_base_` (intermediários `exp_oamil_voc_<scen>.py` e
`exp_vcnc_gate_oamil_full_<scen>.py`). Nenhum override de path
necessário antes de rodar.

A única variável de ambiente obrigatória é
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (sem ela, `randomness.deterministic=True`
faz o PyTorch abortar na primeira operação cuBLAS).

## 2. O que cada config faz

| Config | Cenário | Dataset (.py) | Seed | Schedule | Checkpoint |
|---|---|---|---|---|---|
| `ablation/asym40_loc40/vcnc_oamil_seed2025.py` | asym40 + loc40 | `voc0712_daniel_loc40_asym40.py` | 2025 | 12 epochs, val cada epoch | desabilitado |
| `ablation/asym40_loc40/vcnc_oamil_seed42.py`   | idem | idem | 42 | idem | idem |
| `ablation/loc40/vcnc_oamil_seed2025.py` | loc40 only | `voc0712_daniel_loc40.py` | 2025 | idem | idem |
| `ablation/loc40/vcnc_oamil_seed42.py`   | idem | idem | 42 | idem | idem |

Cadeia de herança:

```
ablation/<scen>/vcnc_oamil_seed<N>.py
    → exp_vcnc_gate_oamil_full_<scen>.py     (intermediário VCNC×OA-MIL)
        → exp_oamil_voc_<scen>.py            (intermediário OA-MIL)
            → _base_/datasets/voc0712_daniel_<dataset>.py
            + _base_/models/faster-rcnn_r50_fpn_20c.py
            + _base_/default_runtime.py
```

Todos compartilham as mesmas hiperparams do gate FIXED (action='filter',
K=30, E1/E4 off, E2/E3 on, gate aggressive_only, progressive_epochs=4)
e do OA-MIL completo (OA-IS na epoch 2, loss_oais com λ=0.1, OA-IE na
epoch 9 com num=4 refine).

## 3. Comandos de invocação

Cada run leva **~1.5–2h** com `deterministic=True` + validação a cada
epoch + OA-IE ativando na epoch 9 (segundo forward extra × 4 iters).
Total estimado para os **4 runs sequenciais**: **~6–8h** numa única
RTX 5090.

Para rodar em background sem prender o terminal, escolher **uma** das
opções abaixo por run.

### Run A — asym40_loc40, seed=2025

```bash
cd /home/pesquisador/pesquisa/filipe/vcnc_mil

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    python tools/train.py \
        custom_configs/exp_configs/ablation/asym40_loc40/vcnc_oamil_seed2025.py \
        --work-dir work_dirs/ablation/asym40_loc40_seed2025 \
        > work_dirs/ablation/asym40_loc40_seed2025.stdout 2>&1 &
```

### Run B — asym40_loc40, seed=42

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    python tools/train.py \
        custom_configs/exp_configs/ablation/asym40_loc40/vcnc_oamil_seed42.py \
        --work-dir work_dirs/ablation/asym40_loc40_seed42 \
        > work_dirs/ablation/asym40_loc40_seed42.stdout 2>&1 &
```

### Run C — loc40, seed=2025

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    python tools/train.py \
        custom_configs/exp_configs/ablation/loc40/vcnc_oamil_seed2025.py \
        --work-dir work_dirs/ablation/loc40_seed2025 \
        > work_dirs/ablation/loc40_seed2025.stdout 2>&1 &
```

### Run D — loc40, seed=42

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    python tools/train.py \
        custom_configs/exp_configs/ablation/loc40/vcnc_oamil_seed42.py \
        --work-dir work_dirs/ablation/loc40_seed42 \
        > work_dirs/ablation/loc40_seed42.stdout 2>&1 &
```

### Alternativa com `tmux` (recomendado para 4 runs sequenciais)

```bash
tmux new -s ablation
# dentro do tmux, executar UM run de cada vez (4× ~1h = ~4h total na mesma GPU)
# ou abrir 4 painéis se houver GPU suficiente (não é o caso aqui — 1 RTX 5090).

# Sequencial em um único painel:
for cfg in \
    custom_configs/exp_configs/ablation/asym40_loc40/vcnc_oamil_seed2025.py \
    custom_configs/exp_configs/ablation/asym40_loc40/vcnc_oamil_seed42.py \
    custom_configs/exp_configs/ablation/loc40/vcnc_oamil_seed2025.py \
    custom_configs/exp_configs/ablation/loc40/vcnc_oamil_seed42.py ; do
    name=$(basename $(dirname $cfg))_$(basename $cfg .py)
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        python tools/train.py $cfg \
            --work-dir work_dirs/ablation/$name \
            2>&1 | tee work_dirs/ablation/$name.stdout
done
```

Sair do tmux com `Ctrl-B D`, voltar com `tmux attach -t ablation`.

## 4. Tabela de resultados (a preencher)

| Cenário | Seed | mAP final (epoch 12) | mAP melhor epoch |
|---|---|---|---|
| asym40_loc40 | 2025 |  |  |
| asym40_loc40 | 42   |  |  |
| loc40        | 2025 |  |  |
| loc40        | 42   |  |  |

Comparar contra:

| Referência | asym40_loc40 mAP | loc40 mAP |
|---|---|---|
| Baseline puro |  |  |
| VCNC sozinho (gate FIXED) |  |  |
| OA-MIL sozinho |  |  |
| **VCNC + OA-MIL (este experimento)** | preencher acima | preencher acima |

## 5. Extração de mAP por run

mmdet 3.x com `VOCMetric` loga `pascal_voc/mAP` (e métricas por classe)
no arquivo `.log` e no `vis_data/scalars.json` de cada work_dir.

### mAP final (última epoch, número 12)

```bash
WD=work_dirs/ablation/asym40_loc40_seed2025  # ajustar por run
grep -oE 'pascal_voc/mAP: [0-9]+\.[0-9]+' "$WD"/*/[0-9]*.log | tail -1
```

### mAP melhor epoch

```bash
WD=work_dirs/ablation/asym40_loc40_seed2025
grep -oE 'pascal_voc/mAP: [0-9]+\.[0-9]+' "$WD"/*/[0-9]*.log \
    | sort -t: -k2 -n | tail -1
```

### mAP por epoch (auditoria completa)

```bash
WD=work_dirs/ablation/asym40_loc40_seed2025
grep -E "Epoch\(val\) \[[0-9]+\]" "$WD"/*/[0-9]*.log \
    | grep -oE '(Epoch\(val\) \[[0-9]+\])|pascal_voc/mAP: [0-9]+\.[0-9]+'
```

### One-liner que preenche a tabela para os 4 runs

```bash
cd /home/pesquisador/pesquisa/filipe/vcnc_mil
for name in asym40_loc40_seed2025 asym40_loc40_seed42 loc40_seed2025 loc40_seed42 ; do
    log=$(ls work_dirs/ablation/$name/*/[0-9]*.log 2>/dev/null | tail -1)
    if [ -z "$log" ]; then
        printf "%-28s | (no log yet)\n" "$name"
        continue
    fi
    final=$(grep -oE 'pascal_voc/mAP: [0-9]+\.[0-9]+' "$log" | tail -1 | awk '{print $2}')
    best=$(grep -oE 'pascal_voc/mAP: [0-9]+\.[0-9]+' "$log" | awk '{print $2}' | sort -n | tail -1)
    printf "%-28s | final=%s | best=%s\n" "$name" "$final" "$best"
done
```

Cole o output diretamente na tabela acima.

### Se a chave for diferente

Se `pascal_voc/mAP` não aparecer (o VOCMetric às vezes loga só `mAP`),
grep alternativo:

```bash
grep -oE '\bmAP\b[: ]+[0-9]+\.[0-9]+' "$WD"/*/[0-9]*.log | tail -1
```

Inspecionar o `.log` rapidamente para confirmar o nome:

```bash
grep -E "mAP" "$WD"/*/[0-9]*.log | head -5
```

## 6. Sanity check rápido (antes de dormir e deixar rodando)

Duas janelas de checagem: a imediata (5min) confirma que o run subiu
sem crash; a estendida (30–40min) confirma que o pipeline OA-MIL
acordou na epoch 2.

### 6a. Checagem imediata (após ~5min, ainda na epoch 1)

```bash
WD=work_dirs/ablation/asym40_loc40_seed2025

# 1. Confirmar que VCNC entrou em warmup (epoch 1, warmup_epochs=1).
grep "Warmup, pulando" "$WD"/*/[0-9]*.log

# 2. Sem crash, sem NaN.
grep -iE "nan|RuntimeError|Traceback|Error" "$WD"/*/[0-9]*.log

# 3. GPU está sendo usada de fato (esperado ~10–20GB ocupados,
#    GPU-Util > 80% no momento de um batch).
nvidia-smi
```

Se (1) imprime a linha de warmup, (2) volta vazio, e (3) mostra
utilização da GPU — pode soltar os outros runs.

### 6b. Checagem estendida (após ~30–40min, já na epoch 2)

```bash
WD=work_dirs/ablation/asym40_loc40_seed2025

# 1. Sanity log do OA-MIL disparou (uma vez, no primeiro batch da epoch 2).
#    n_oaie_iters=1 NESTA fase porque oaie_epoch=9; só vai mudar pra 5
#    a partir da epoch 9 — mas o sanity já foi printado uma vez e não
#    aparece de novo (flag self._sanity_printed).
grep "OA-MIL sanity" "$WD"/*/[0-9]*.log

# 2. loss_oais começou a aparecer no log de treino (epoch 2+).
grep -E "loss_oais" "$WD"/*/[0-9]*.log | head -5

# 3. VCNC rodou E2 (clustering) e E3 (spatial) na epoch 2 — gate ainda
#    em fase progressiva (progressive_epochs=4 → gate só na epoch 5).
grep -E "ETAPA 2|ETAPA 3|Época 2" "$WD"/*/[0-9]*.log | head -10
```

Se (1) imprime a linha de sanity com os 4 `requires_grad` esperados,
(2) mostra valores de `loss_oais` (~0.07–0.10 inicialmente), e (3)
mostra os banners de ETAPA 2/3 sob a Época 2 — está tudo conforme.

## 7. Limpeza opcional após coleta dos números

Como `default_hooks.checkpoint=None`, nenhum `.pth` é criado. Os
`work_dirs/ablation/*/` ficam pequenos (~MB de log + scalars JSON).
Pode arquivar:

```bash
tar czf ablation_logs_$(date +%Y%m%d).tar.gz work_dirs/ablation/
```

---

**Nota sobre OA-IE:** ativa só na **epoch 9** (paper VOC). Antes disso,
`n_oaie_iters=1` no sanity log (que foi impresso uma única vez na
epoch 2, antes do OA-IE estar disponível — não é re-printado quando
OA-IE liga). Para confirmar que OA-IE entrou em ação, observe a
mudança no valor de `loss_oais` ao cruzar a epoch 9 — provável
**aumento súbito** porque a fórmula passa de `(1 − s₀)·λ` (Step 3)
para `((1 − s₀) + (1 − mean(s₁..s₄))·oaie_coef)·λ` (Step 4 refine),
adicionando um segundo termo positivo. Esse salto é esperado e não
indica bug.
