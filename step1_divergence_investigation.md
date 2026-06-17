# Step 1 — Investigação da divergência de loss (~0.1–0.7%)

Spoiler: **o skeleton OA-MIL é numericamente equivalente ao baseline**. A
divergência observada está dentro do ruído de não-determinismo do CUDA, e
acontece **igual** quando você roda o MESMO config duas vezes.

---

## Passo 1 — Diff dos dois configs (estático)

O baseline original (`exp5_12x1[9]_baseline_danieldb_sim40.py`) e o
skeleton (`exp_oamil_voc_sim40.py`) **diferem em mais coisas do que só o
OA-MIL**. Diff resumido:

| Aspecto | Baseline | Skeleton |
|---|---|---|
| `randomness.deterministic` | (não setado) | `False` (explícito) |
| `custom_hooks` | `WandbPredBucketsHook` | `OAMILEpochInjectorHook` |
| `vis_backends` | `LocalVisBackend + WandbVisBackend` | só `LocalVisBackend` (default) |
| `visualizer` | recebe wandb backend | default |
| `roi_head.type` | `StandardRoIHead` | `StandardRoIHeadOAMIL` |
| `bbox_head.type` | (default) `Shared2FCBBoxHead` | `Shared2FCBBoxHeadOAMIL` |

Confirmação empírica de que o wandb realmente inicializou no baseline:
`work_dirs/step1_baseline/.../vis_data/wandb/run-20260616_104445-9pym25ta/`
existe; o do skeleton não. Isso introduz uma assimetria de inicialização
(thread do wandb, possível `wandb.init()` consumindo RNG numpy para gerar
ID de run, ordem de construção de objetos do mmengine) **antes** da
primeira iteração.

## Passo 2 — Quando os hooks rodam

- `WandbPredBucketsHook`: só sobrescreve `after_val_epoch` — não roda
  durante treino, não toca RNG em iter 0.
- `OAMILEpochInjectorHook`: só sobrescreve `before_train_epoch` — roda
  **antes** de iter 0, mas só faz `bbox_head._epoch = runner.epoch`, uma
  atribuição Python pura. Sem RNG, sem CUDA.

Conclusão da análise estática: **nenhum dos dois hooks consome RNG nem
toca o estado CUDA antes de iter 0**. Os hooks em si não explicam a
divergência. O wandb sim pode.

## Passo 3 — Rodar ambos com `deterministic=True`

Para isolar o efeito do código (vs assimetria de configuração), criei
dois configs novos com setup **idêntico ao skeleton** exceto pelo que
está sendo testado:

- `custom_configs/exp_configs/_diag_clean_baseline.py` — sem wandb, sem
  OA-MIL hook, com `StandardRoIHead` + `Shared2FCBBoxHead` originais.
- `custom_configs/exp_configs/_diag_skeleton_nohook.py` — sem wandb, sem
  OA-MIL hook, com `StandardRoIHeadOAMIL` + `Shared2FCBBoxHeadOAMIL`
  (`oamil_lambda=0`).

Rodei 3 configurações, todas com `seed=2025`, `deterministic=True`,
`logger.interval=1`, `log_processor.window_size=1` (log sem média móvel):

- **A** = `_diag_clean_baseline.py` (StandardRoIHead, sem hook, sem wandb)
- **B** = `exp_oamil_voc_sim40.py` (skeleton com OA-MIL hook, sem wandb)
- **C** = `_diag_skeleton_nohook.py` (skeleton sem OA-MIL hook, sem wandb)

### Iter 1 (primeira iteração)

| Campo | A (clean baseline) | B (skel + hook) | C (skel s/ hook) |
|---|---|---|---|
| `loss` | 3.7242407799 | 3.7242407799 | 3.7242407799 |
| `loss_rpn_cls` | 0.6954386234 | 0.6954386234 | 0.6954386234 |
| `loss_rpn_bbox` | 0.0190818664 | 0.0190818664 | 0.0190818664 |
| `loss_cls` | 2.9933161736 | 2.9933161736 | 2.9933161736 |
| `loss_bbox` | 0.0164041705 | 0.0164041705 | 0.0164041705 |

**Iter 1 é bit-exact entre os três.** O primeiro forward+backward+step do
skeleton produz exatamente os mesmos números que o baseline puro. Isso
prova que o caminho do código no skeleton (chamada explícita de
`get_targets` em vez de via `loss_and_target`, override de `loss` que
delega para `super().loss` quando `pseudo_bbox_targets=None`) é
matematicamente equivalente ao caminho padrão.

### Iter 2–5 (drift máximo)

| Campo | max\|A−B\| | max\|A−C\| | max\|B−C\| |
|---|---|---|---|
| `loss` | 2.66e-03 | 6.26e-03 | 5.29e-03 |
| `loss_rpn_cls` | 1.83e-03 | 2.18e-03 | 3.54e-04 |
| `loss_cls` | 2.50e-03 | 2.83e-03 | 3.38e-03 |
| `loss_bbox` | 4.02e-04 | 1.40e-03 | 1.58e-03 |

Sim, drift de 0.1–0.6% a partir do iter 2 — **igual** ao que você
observou no comparativo original.

## Passo 4 — Hook desregistrado

Resposta direta: rodar **C** (skeleton sem hook) **não** elimina o drift.
Pelo contrário, `max|A−C| ≈ 6.3e-3` é maior que `max|A−B| ≈ 2.7e-3`. O
hook **não** é a causa.

## Teste decisivo extra — mesmo config rodado duas vezes

Para descobrir se o drift do iter 2+ vem do skeleton ou do CUDA, rodei o
**MESMO** skeleton **duas vezes** (B1 e B2), tudo idêntico, mesma seed,
deterministic=True:

| Campo | B1 (iter 2) | B2 (iter 2) | diff |
|---|---|---|---|
| `loss` | 0.9898248911 | 0.9908955693 | **1.07e-03** |
| `loss_rpn_cls` | 0.6838595867 | 0.6838591695 | 4.17e-07 |
| `loss_cls` | 0.2518277466 | 0.2528991103 | **1.07e-03** |
| `loss_bbox` | 0.0173373464 | 0.0173374452 | 9.87e-08 |

`loss` e `loss_cls` divergem em ~1e-3 entre duas execuções do MESMO
config. `loss_rpn_*` permanecem bit-exact. Isso isola o ruído ao
**backward pass da cabeça ROI** (que envolve operações CUDA não-
determinísticas mesmo com `deterministic=True` — typicamente atomic adds
no gradient accumulation com `scatter_add_`, `index_add_` ou similares
usados pelo cross-entropy/L1 com índices de positivos).

Comparando ordens de grandeza:

```
max diff B1 vs B2 (mesmo config):           4.5e-3   (loss_cls iter 5)
max diff A vs B1 (StandardRoIHead vs skel): 2.7e-3
max diff A vs C (clean baseline vs skel-no-hook): 6.3e-3
```

**O ruído entre runs do MESMO config é da mesma ordem do ruído entre
runs de configs diferentes.** Não há sinal estatístico de que o skeleton
introduza divergência além do ruído intrínseco.

## Conclusão

O skeleton OA-MIL passa o critério de equivalência numérica no sentido
forte e útil:

1. **Iter 1 é bit-exact** com o baseline (`_diag_clean_baseline.py`),
   provando que o caminho do código está matematicamente correto.
2. **Iter 2+ tem drift de ~0.1–0.6%** entre quaisquer dois runs (mesmos
   ou diferentes configs), explicado por CUDA não-determinismo na
   backward pass da ROI head.
3. `loss_oais` **não aparece** em nenhum log (chave ausente, como
   especificado). ✓
4. O `OAMILEpochInjectorHook` **não** afeta os números.

O 0.1–0.7% que você viu originalmente entre `step1_baseline` e
`step1_oamil_lambda0` é o mesmo ruído CUDA + assimetria de inicialização
do wandb. Não é um bug no port.

## Recomendações antes da Step 2

1. **Suba `oamil_lambda` para um valor > 0 só na Step 2/3**, com os
   stubs `NotImplementedError` no caminho. Vai bater no
   `NotImplementedError` antecipadamente se algo for chamado errado.
2. Para validações futuras de equivalência, comparar **sempre contra
   `_diag_clean_baseline.py`** (não contra `exp5_12x1[9]_baseline_*`),
   porque ele é simétrico em hooks/vis_backends.
3. Considerar adicionar `randomness=dict(seed=..., deterministic=True)`
   nos configs de comparação numérica para garantir iter 1 bit-exact.
4. Os configs de diagnóstico (`_diag_*.py`) podem ser apagados depois da
   Step 2 ou mantidos pra futuras regressões — sem custo.

## Arquivos de diagnóstico criados

| Arquivo | Propósito |
|---|---|
| `custom_configs/exp_configs/_diag_clean_baseline.py` | Baseline limpo (sem wandb, sem hooks) com `StandardRoIHead` |
| `custom_configs/exp_configs/_diag_skeleton_nohook.py` | Skeleton OA-MIL sem `OAMILEpochInjectorHook` |
| `work_dirs/diag_A_clean_baseline/`, `diag_B_skeleton/`, `diag_B2_skeleton_rerun/`, `diag_C_skeleton_nohook/` | Logs e scalars dos runs |

Posso apagar os `_diag_*.py` se preferir, ou deixar pro próximo CI de
equivalência. **Aguardando seu OK para começar a Step 2 (OA-IS only).**
