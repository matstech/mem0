# Benchmark Comparison Tool

Questo tool permette di confrontare diverse esecuzioni di benchmark (baseline) producendo report tabellari, grafici e un PDF riassuntivo.

## Requisiti

Il tool utilizza Python 3 e le seguenti librerie:
- `pandas`
- `matplotlib`

## Utilizzo CLI

Il punto di ingresso è `comparison_summary/run_comparison.py`.

```bash
python3 comparison_summary/run_comparison.py \
    --comparison-name "nome_confronto" \
    --baseline "Label,results_path,metrics_path,token_accounting_path" \
    --baseline "AltraLabel,results_path,metrics_path" \
    --notes "Note opzionali"
```

### Formato Argomento --baseline
L'argomento `--baseline` accetta una stringa separata da virgole con i seguenti campi (in ordine):
1. **Label**: Nome visualizzato nel report (es. `mem0_local`).
2. **Results Path**: Percorso al file `*_results.json`.
3. **Metrics Path** (Opzionale): Percorso al file `*_eval_metrics.json`.
4. **Token Accounting Path** (Opzionale): Percorso al file `*_token_accounting_*.json`.

## Output Generati

I risultati vengono salvati in `results/comparison_summary/<nome>_<timestamp>/`:
- `summary.md`: Riepilogo rapido in Markdown.
- `comparison_report.pdf`: Report completo con tabelle e grafici.
- `tables/`: CSV delle comparazioni (overall, per categoria, dettagli).
- `figures/`: Grafici PNG (Qualità, Latenza, Token, Trade-off).
- `comparison_manifest.json`: Metadati del confronto.
- `<nome>_<timestamp>.zip`: Archivio compresso di tutti gli artifact.

## Integrazione con Makefile

Per semplificare l'uso, sono disponibili dei target nel `Makefile` della root.
