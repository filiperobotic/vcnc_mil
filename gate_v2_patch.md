# Patch: gate de confusão v2 (COCO-safe) — `vcnc_kmeans_confusion_aware_spatial_hook.py`

Todos os novos parâmetros têm default que reproduz o comportamento atual.
Configs do VOC não precisam mudar.

---

## 1. `__init__` — adicionar 3 parâmetros após `confusion_gate_ratio_factor`

```python
                 confusion_gate_ratio_factor: float = 10.0,
                 # === GATE v2 (NEW) — defaults reproduzem o comportamento antigo ===
                 # 'mean': sev = (C_ij + C_ji)/2   (comportamento original)
                 # 'min' : sev = min(C_ij, C_ji)   (exige reciprocidade — elimina
                 #         pares person<->X gerados por ruído simétrico + desbalanceamento)
                 confusion_gate_severity: str = 'mean',
                 # Piso absoluto do threshold. 0.0 = sem piso (original).
                 # Evita threshold ~0 quando a distribuição de severidade é zero-inflada
                 # (80 classes -> ~3000 pares, quase todos com confusão 0).
                 confusion_gate_min_threshold: float = 0.0,
                 # Se True, mediana/MAD são calculados só sobre pares com sev > 0.
                 confusion_gate_ignore_zero_pairs: bool = False,
```

E no corpo do `__init__`, logo após `self.confusion_gate_ratio_factor = ...`:

```python
        self.confusion_gate_severity = confusion_gate_severity
        assert self.confusion_gate_severity in ('mean', 'min'), \
            f"confusion_gate_severity deve ser 'mean' ou 'min', recebeu {confusion_gate_severity!r}"
        self.confusion_gate_min_threshold = confusion_gate_min_threshold
        self.confusion_gate_ignore_zero_pairs = confusion_gate_ignore_zero_pairs
```

---

## 2. `_compute_confusion_signals` — substituir dois blocos

### 2a. Cálculo da severidade (dentro do loop `for j in range(i + 1, K)`)

Trocar:
```python
                severity = 0.5 * (C[i, j] + C[j, i])
```
por:
```python
                if self.confusion_gate_severity == 'min':
                    severity = min(C[i, j], C[j, i])
                else:
                    severity = 0.5 * (C[i, j] + C[j, i])
```

### 2b. Threshold (do `severities = np.array(...)` até `threshold = max(...)`)

Trocar:
```python
        severities = np.array([s for _, _, s in sev_pairs], dtype=np.float64)
        median_sev = float(np.median(severities))
        mad_sev = float(np.median(np.abs(severities - median_sev)))

        thr_mad = median_sev + self.confusion_gate_mad_factor * mad_sev
        thr_ratio = self.confusion_gate_ratio_factor * median_sev
        threshold = max(thr_mad, thr_ratio)
```
por:
```python
        severities = np.array([s for _, _, s in sev_pairs], dtype=np.float64)
        ref = severities[severities > 0] if self.confusion_gate_ignore_zero_pairs else severities
        if ref.size == 0:
            ref = severities  # todos zero: cai no piso absoluto abaixo
        median_sev = float(np.median(ref))
        mad_sev = float(np.median(np.abs(ref - median_sev)))

        thr_mad = median_sev + self.confusion_gate_mad_factor * mad_sev
        thr_ratio = self.confusion_gate_ratio_factor * median_sev
        threshold = max(thr_mad, thr_ratio, self.confusion_gate_min_threshold)
```

### 2c. (opcional, só log) adicionar ao dict `stats`:
```python
            'n_nonzero_pairs': int((severities > 0).sum()),
            'severity_mode': self.confusion_gate_severity,
            'min_threshold': self.confusion_gate_min_threshold,
```
e no log de debug em `before_train_epoch`, após a linha do Threshold:
```python
                self._logger.info(f"[VCNC-Spatial] Severidade modo={confusion_stats['severity_mode']}, "
                      f"pares com sev>0: {confusion_stats['n_nonzero_pairs']}, "
                      f"piso={confusion_stats['min_threshold']:.3f}")
```

---

## 3. Config COCO — adicionar ao dict do hook

```python
        confusion_gate_severity='min',
        confusion_gate_min_threshold=0.05,
        confusion_gate_ignore_zero_pairs=True,
```

---

## 4. Runs de diagnóstico (ordem sugerida)

| # | Cenário | Config | Objetivo |
|---|---|---|---|
| A | sim40 nc120 | `confusion_gate_action='skip'` (resto original) | Teto: quanto o gate estava destruindo |
| B | sim40 nc120 | gate v2 (`min`, piso 0.05, ignore_zero) + `action='filter'` | Gate corrigido resolve? |
| C | loc20 nc120 | gate v2 + `action='filter'` | Pares marcados deve cair de ~600 para ~0 |
| D | loc40 nc120 | **rerodar** com `instances_train2017_shift_40.json` confirmado no log, e antes checar boxes degenerados no json | O run nc90 anterior era loc20 |

Verificação rápida no log de C: `Pares marcados como ruidosos:` deve ficar próximo de 0
e `filter=` na linha `Gate (action=...)` deve cair para poucos milhares ou zero.
Em B, os top pares (0, X) com C[0->X]=0 não devem mais aparecer como ★ RUIDOSO.
