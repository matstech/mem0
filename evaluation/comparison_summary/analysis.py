import pandas as pd
from pathlib import Path
from .utils import read_json

def normalize_baseline_data(baseline_info):
    """
    baseline_info: {
        'label': str,
        'results_path': str,
        'metrics_path': str (optional),
        'token_accounting_path': str (optional),
        'category_mapping': dict (optional)
    }
    """
    label = baseline_info['label']
    results = read_json(baseline_info['results_path'])
    metrics = read_json(baseline_info.get('metrics_path'))
    token_accounting = read_json(baseline_info.get('token_accounting_path'))
    
    if results is None:
        return None
        
    # Flatten results
    rows = []
    for conv_id, items in results.items():
        for item in items:
            search_time = item.get('search_time')
            if search_time is None:
                speaker_1_time = item.get('speaker_1_memory_time')
                speaker_2_time = item.get('speaker_2_memory_time')
                if speaker_1_time is not None and speaker_2_time is not None:
                    search_time = speaker_1_time + speaker_2_time
            row = {
                'baseline': label,
                'conversation_id': conv_id,
                'question': item.get('question'),
                'category': str(item.get('category')),
                'search_time': search_time,
                'response_time': item.get('response_time'),
                'context_tokens': item.get('context_tokens'),
                'prompt_tokens_total': item.get('prompt_tokens_total'),
                'total_tokens': item.get('total_tokens'),
                'total_pipeline_tokens_amortized': item.get('total_pipeline_tokens_amortized'),
                'recalled_count': item.get('recalled_count'),
            }
            
            # Try to enrich with metrics if available
            if metrics and conv_id in metrics:
                # Metrics might be a list of dicts for this conv_id
                # Find the one matching the question
                for metric_item in metrics[conv_id]:
                    if metric_item.get('question') == row['question']:
                        row.update({
                            'bleu_score': metric_item.get('bleu_score'),
                            'f1_score': metric_item.get('f1_score'),
                            'llm_score': metric_item.get('llm_score'),
                        })
                        break
            
            rows.append(row)
            
    df = pd.DataFrame(rows)
    
    # Enrich with canonical tokens from token_accounting if available
    token_summary = None
    if token_accounting:
        token_summary = {
            'run_total_tokens': token_accounting.get('run_total_tokens'),
            'ingest_total_tokens': token_accounting.get('run_token_breakdown', {}).get('ingest', {}).get('total_tokens'),
            'retrieval_total_tokens': token_accounting.get('run_token_breakdown', {}).get('retrieval', {}).get('total_tokens'),
            'answer_total_tokens': token_accounting.get('run_token_breakdown', {}).get('answer', {}).get('total_tokens'),
            'maintenance_total_tokens': token_accounting.get('run_token_breakdown', {}).get('maintenance', {}).get('total_tokens'),
        }
    
    return {
        'df': df,
        'token_summary': token_summary,
        'info': baseline_info
    }

def build_overall_comparison(normalized_baselines):
    summary_rows = []
    for entry in normalized_baselines:
        df = entry['df']
        token_summary = entry['token_summary']
        info = entry['info']
        
        row = {
            'baseline': entry['info']['label'],
            'BLEU': df['bleu_score'].mean() if 'bleu_score' in df else None,
            'F1': df['f1_score'].mean() if 'f1_score' in df else None,
            'LLM Judge': df['llm_score'].mean() if 'llm_score' in df else None,
            'search_time': df['search_time'].mean() if 'search_time' in df else None,
            'response_time': df['response_time'].mean() if 'response_time' in df else None,
            'context_tokens': df['context_tokens'].mean() if 'context_tokens' in df else None,
            'prompt_tokens_total': df['prompt_tokens_total'].mean() if 'prompt_tokens_total' in df else None,
            'total_pipeline_tokens_amortized': df['total_pipeline_tokens_amortized'].mean() if 'total_pipeline_tokens_amortized' in df else None,
        }
        
        question_level_total_tokens = df['total_tokens'].mean() if 'total_tokens' in df else None
        run_total_tokens = token_summary['run_total_tokens'] if token_summary else None
        row['run_total_tokens'] = run_total_tokens
        if question_level_total_tokens is not None and pd.notna(question_level_total_tokens):
            row['total_tokens'] = question_level_total_tokens
            row['token_metric_status'] = 'canonical_question_mean'
        elif run_total_tokens is not None:
            row['total_tokens'] = run_total_tokens
            row['token_metric_status'] = 'canonical_run'
        else:
            row['total_tokens'] = None
            row['run_total_tokens'] = None
            row['token_metric_status'] = 'legacy_only' if row['total_pipeline_tokens_amortized'] is not None else 'missing'
            
        summary_rows.append(row)
        
    return pd.DataFrame(summary_rows)

def build_per_category_comparison(normalized_baselines):
    all_dfs = []
    for entry in normalized_baselines:
        df = entry['df']
        if df.empty:
            continue
            
        agg_map = {'question': 'count'}
        for col in ['bleu_score', 'f1_score', 'llm_score']:
            if col in df.columns:
                agg_map[col] = 'mean'
                
        cat_agg = df.groupby('category').agg(agg_map).rename(columns={'question': 'count'}).reset_index()
        cat_agg['baseline'] = entry['info']['label']
        all_dfs.append(cat_agg)
        
    if not all_dfs:
        return pd.DataFrame()
        
    return pd.concat(all_dfs, ignore_index=True)

def build_per_baseline_details(normalized_baselines):
    details = []
    for entry in normalized_baselines:
        info = entry['info']
        details.append({
            'baseline': info['label'],
            'source_result_path': info['results_path'],
            'source_metrics_path': info.get('metrics_path'),
            'source_token_accounting_path': info.get('token_accounting_path'),
            'notes': info.get('notes', '')
        })
    return pd.DataFrame(details)
