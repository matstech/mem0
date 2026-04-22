# Benchmark Comparison Tool

Questo tool permette di confrontare diverse esecuzioni di benchmark (baseline) producendo report tabellari, grafici e un PDF riassuntivo.

## Requisiti

Il tool utilizza Python 3 e le seguenti librerie:
- `pandas`
- `matplotlib`

## Utilizzo tramite Makefile (Consigliato)

È possibile utilizzare il `Makefile` per lanciare il confronto in modo semplificato.

### Comando Base
```bash
make compare BASELINES="mem0_local,path/to/mem0_local_results.json,path/to/mem0_local_metrics.json rag,path/to/rag_results.json,path/to/rag_metrics.json"
```

### Esempio con Tutte le Variabili
```bash
make compare \
    NAME="subset2_analysis" \
    BASELINES="mem0_local,results/subset2/mem0_local_results.json,results/subset2/mem0_local_metrics.json rag,results/subset2/rag_results.json,results/subset2/rag_metrics.json" \
    NOTES="Analisi comparativa su subset2"
```

*Nota: Le baseline nella variabile `BASELINES` devono essere separate da uno spazio.*

## Utilizzo tramite CLI (Avanzato)

Se preferisci usare direttamente lo script Python:

```bash
python3 comparison_summary/run_comparison.py \
    --comparison-name "nome_confronto" \
    --baseline "Label1,results_path,metrics_path,token_path" \
    --baseline "Label2,results_path" \
    --notes "Note opzionali"
```

## Struttura degli Output

I risultati vengono salvati in `results/comparison_summary/<nome>_<timestamp>/`:
- `summary.md`: Riepilogo rapido.
- `comparison_report.pdf`: Report completo pronto per la consultazione.
- `tables/`: CSV dettagliati.
- `figures/`: Grafici PNG.
- `<nome>_<timestamp>.zip`: Archivio compresso di tutto il run.

## Altri Target Utili
- `make clean-comparisons`: Rimuove tutti i risultati delle comparazioni precedenti.
