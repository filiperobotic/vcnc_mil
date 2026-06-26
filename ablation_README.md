# Ablation Sweep — VCNC gate × OA-MIL

44 runs (22 cenários × 2 seeds), todos com gate VCNC (config FIXED) +
OA-MIL completo (paper VOC). Orquestrados por
`custom_configs/exp_configs/ablation/run_ablation_all.sh`.

Os baselines de comparação (VCNC sozinho, OA-MIL sozinho, sem ruído)
**não** estão neste sweep — já existem de experimentos anteriores.

## 1. Pré-requisitos

**Ative o conda env** antes de invocar o script:

```bash
conda activate mmlab-bw311   # ou o nome do env que tem mmdet 3.x + mmengine
```

O script aborta cedo (`[ABORT] python env sem mmdet`) se `import mmdet`
falhar — ele NÃO ativa env nenhum sozinho. A variável
`CUBLAS_WORKSPACE_CONFIG=:4096:8` é exportada pelo script automaticamente.

## 2. Invocação

```bash
cd /home/pesquisador/pesquisa/filipe/vcnc_mil

# Recomendado: dentro de tmux pra sobreviver a desconexões SSH
tmux new -s ablation

bash custom_configs/exp_configs/ablation/run_ablation_all.sh \
     2>&1 | tee custom_configs/exp_configs/ablation/orchestrator.log
```

Sair do tmux com `Ctrl-B D`, voltar com `tmux attach -t ablation`.

O script faz 3 coisas antes do primeiro `python tools/train.py`:

1. Verifica que `mmdet` está importável no Python ativo.
2. Cria/abre `ablation_progress.txt` (header CSV se primeiro run).
3. Valida os 44 configs via `Config.fromfile` (catches missing files,
   broken `_base_` chains, dataset references quebradas).

Se a validação falhar, **nenhum** treino é iniciado.

## 3. Resumir / retomar

O script é resumável. Se for matado mid-sweep ou se algum run falhar,
basta rerodar o mesmo comando:

```bash
bash custom_configs/exp_configs/ablation/run_ablation_all.sh
```

Para cada `(scenario, seed)`, o script:

- Verifica se `ablation_progress.txt` já tem uma linha
  `<scen>,<seed>,DONE,...`. Se sim, imprime `[SKIP] ...` e pula.
- Caso contrário, roda do zero (sobreescreve `work_dirs/ablation/<tier>_<scen>_seed<S>/`).

Para forçar re-execução de um run específico, editar
`ablation_progress.txt` e remover a linha correspondente, OU deletar
o work_dir e o status no progress file.

## 4. Acompanhar progresso em tempo real

```bash
# Tail do progress file (CSV)
tail -f custom_configs/exp_configs/ablation/ablation_progress.txt

# Tail do orchestrator log (stdout do script)
tail -f custom_configs/exp_configs/ablation/orchestrator.log

# Logs detalhados do run atual (substitua <name> apropriadamente)
tail -f work_dirs/ablation/tier1_loc20_asym20_seed2025.stdout
```

## 5. Descobrir runs que falharam

Tudo que NÃO é `DONE` aparece com este comando (mantém o header
e expõe TIMEOUT, FAILED_exit*, MISSING_CONFIG):

```bash
grep -v ",DONE," custom_configs/exp_configs/ablation/ablation_progress.txt
```

Tipos de status possíveis:
- `DONE` — run completou em ≤3h, mAP extraído.
- `TIMEOUT` — `timeout 3h` disparou (exit code 124).
- `FAILED_exit<N>` — Python abortou com exit code N (NaN, OOM, etc.).
- `MISSING_CONFIG` — só dispara se um config sumir entre validação e
  o main loop (não deveria, mas é defensivo).

Para investigar uma falha específica, abrir
`work_dirs/ablation/<tier>_<scen>_seed<S>.stdout` (stdout do
`tools/train.py`) e o `.log` interno em
`work_dirs/ablation/<tier>_<scen>_seed<S>/<timestamp>/*.log`.

## 6. Estimativa de tempo

- **Por run:** ~1.5h (12 epochs, `deterministic=True`, val a cada epoch,
  OA-IE ativando na epoch 9).
- **Total sequencial:** ~66h (44 × 1.5h) numa única RTX 5090.
- **Timeout por run:** 3h (margem 2× sobre o esperado).

Se a GPU estiver compartilhada com outra coisa, esperar mais. Se algum
run estourar 3h é provavelmente travamento (NaN loop, deadlock CUDA);
verificar o `.stdout` correspondente.

## 7. Extrair tabela consolidada quando terminar

```bash
cd /home/pesquisador/pesquisa/filipe/vcnc_mil

# Tabela completa formatada
column -t -s, custom_configs/exp_configs/ablation/ablation_progress.txt

# Ordenar por mAP_best descendente (skip header + lines NA)
{ head -1 custom_configs/exp_configs/ablation/ablation_progress.txt;
  tail -n +2 custom_configs/exp_configs/ablation/ablation_progress.txt \
    | grep -v ",NA," | sort -t, -k5 -rn ; } \
  | column -t -s,
```

## 8. Estrutura no disco

```
custom_configs/exp_configs/
├── ablation/
│   ├── ablation_progress.txt           ← CSV criado pelo script
│   ├── orchestrator.log                ← stdout do script (via tee)
│   ├── run_ablation_all.sh             ← orquestrador
│   ├── tier1_<scenario>/
│   │   ├── vcnc_oamil_seed2025.py
│   │   └── vcnc_oamil_seed42.py
│   ├── tier2_<scenario>/  ...
│   ├── tier3_<scenario>/  ...
│   └── tier4_<scenario>/  ...
│
├── ablation_intermediates/
│   ├── exp_oamil_voc_<scenario>.py            ← × 22 (OA-MIL only)
│   └── exp_vcnc_gate_oamil_full_<scenario>.py ← × 22 (OA-MIL + VCNC gate)
│
└── exp_*.py                                    ← configs antigos (sim40, etc.)

work_dirs/ablation/
└── <tier>_<scenario>_seed<S>/                  ← criado pelo script
    └── <timestamp>/
        ├── *.log                               ← log do mmengine
        └── vis_data/scalars.json               ← métricas por iter
```

Nenhum `.pth` é gerado (todos os configs setam
`default_hooks=dict(checkpoint=None)`). Os work_dirs ficam pequenos
(~MB de log + JSON por run).
