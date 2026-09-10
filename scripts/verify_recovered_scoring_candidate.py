"""Verify the later retained scoring candidate on SHA-bound frozen reports.

Runs only the AST-isolated pairing function and original lexical/refusal helpers.
The general aggregator is disabled; no model, image generation, checkpoint load,
or bootstrap runs. Exact input/source bytes are verified before interpretation.
Python standard library only. April20 runtime-to-source binding remains unproved.
"""
from pathlib import Path
import argparse
from functools import lru_cache
import ast, collections, hashlib, json, logging, math, sys, types
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts', required=True, type=Path, help='Extracted archive root containing artifacts/Results, or inner root containing Results.')
parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1]/'provenance/later_scoring_source/spinefairbench')
parser.add_argument('--output', type=Path, help='Optional verification JSON; parent directory must exist.')
args=parser.parse_args()
COMMIT='0e114cdca9d522054308c6d74ce14b30c698503a'
SRC=args.source_root
ART=args.artifacts
if not (ART/'Results').is_dir() and (ART/'artifacts/Results').is_dir():
 ART=ART/'artifacts'
EXPECTED={
'analyze.py':'aa29f549177df5314cf27e79aa379ce7a442f05f9017c218b22b91d4d7655756',
'metrics/refusal_detector.py':'06a126ca0b50dfcac6acbae862a90df18430f851419a248a2e923273a47b21f1',
'metrics/hallucination.py':'fe82ca6cbad226bbb47129b3a958ae86dc2bf7e08ae7d0e73d0992a4f9fc3840',
'metrics/diagnostic_label.py':'79650c751cb346c688d0741df7f872b6cafb9076d761f04e6ffa4961702240ea',
'data/annotations.py':'2bed2312045293de07b8196454d2e957e291bce190f759122ad7881be6a9929c'}
INPUT_HASHES={'Results/analysis/common_core_1000_summary.json': 'e0eb42ebcb28b8e8aae16ae7749ae0adb810bf0c0c8f5ca2a8810d50993611f5', 'Results/final_inputs/panels/full_pipeline_retained/evaluation_results.json': '5e3ff8d1c3cc808e2199e1c19f0b6c456437c21db2bc58df05aca0c34891b7b3', 'Results/final_inputs/panels/baseline_only_retained/evaluation_results.json': '7d900f5e2965e9ad863e0c1a0a6e2bcd41a03c5a9bc7339efc60566172b0e67a', 'Results/final_inputs/panels/full_pipeline_retained/pairs.json': 'cb5c29f25c54a086c934c77517e8e28ae695704192f181084c5efdc5a23c58bb', 'Results/final_inputs/panels/baseline_only_retained/pairs.json': 'cb5c29f25c54a086c934c77517e8e28ae695704192f181084c5efdc5a23c58bb', 'freeze_runs/2026-04-09/evaluation_source_subset_core1000.json': '56d52393844c11d9aa5b9aed9fe9db144dfa8193c2584d7d980d501bedc2af4d'}
for relative, expected in INPUT_HASHES.items():
 hasher=hashlib.sha256()
 with (ART/relative).open("rb") as handle:
  for chunk in iter(lambda:handle.read(8*1024*1024),b""):hasher.update(chunk)
 if hasher.hexdigest()!=expected:raise ValueError(f"Input SHA-256 mismatch: {relative}")
source_bytes={}
for p,h in EXPECTED.items():
 b=(SRC/p).read_bytes()
 if hashlib.sha256(b).hexdigest()!=h:raise ValueError(f'Source SHA-256 mismatch: {p}')
 source_bytes[p]=b
for package in ['spinefairbench','spinefairbench.metrics','spinefairbench.data']:
 m=types.ModuleType(package);m.__path__=[];sys.modules[package]=m
for name in ['diagnostic_label','refusal_detector','hallucination']:
 fullname='spinefairbench.metrics.'+name
 mod=types.ModuleType(fullname);sys.modules[fullname]=mod
 exec(compile(source_bytes[f'metrics/{name}.py'],str(SRC/'metrics'/f'{name}.py'),'exec'),mod.__dict__)
annotations=types.ModuleType('spinefairbench.data.annotations')
astnodes=ast.parse(source_bytes['data/annotations.py']).body
cats=next(ast.literal_eval(n.value) for n in astnodes if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ABNORMALITY_CATEGORIES' for t in n.targets))
annotations.INDEX_TO_CATEGORY=dict(enumerate(cats));sys.modules[annotations.__name__]=annotations
aggregator=types.ModuleType('spinefairbench.metrics.aggregator')
aggregator.aggregate_metrics=lambda *args,**kwargs:{}
sys.modules[aggregator.__name__]=aggregator
node=next(n for n in ast.parse(source_bytes['analyze.py']).body if isinstance(n,ast.FunctionDef) and n.name=='_compute_fairness_metrics')
node.returns=None
for arg in node.args.args:arg.annotation=None
ns={'logger':logging.getLogger('recheck')}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact_recovered_pairing_function>','exec'),ns)
summary=json.loads((ART/'Results/analysis/common_core_1000_summary.json').read_bytes())
core=set(json.loads((ART/'freeze_runs/2026-04-09/evaluation_source_subset_core1000.json').read_bytes())['source_ids'])
raw_fpr=sys.modules['spinefairbench.metrics.hallucination'].compute_false_positive_rate
@lru_cache(maxsize=100000)
def cached_fpr(text,ground):
 return raw_fpr(text,set(ground))
def fpr(text,ground):
 return cached_fpr(text,frozenset(ground))
sys.modules['spinefairbench.metrics.hallucination'].compute_false_positive_rate=fpr
refusal=sys.modules['spinefairbench.metrics.refusal_detector']
refusal.classify_response=lru_cache(maxsize=100000)(refusal.classify_response)
results={}
inputs={}
for panel,short in [('full_pipeline_retained','full'),('baseline_only_retained','baseline')]:
 folder=ART/'Results/final_inputs/panels'/panel
 for fname in ['pairs.json','evaluation_results.json']:inputs[f'Results/final_inputs/panels/{panel}/{fname}']=hashlib.sha256((folder/fname).read_bytes()).hexdigest()
 pairs=json.loads((folder/'pairs.json').read_bytes());rows=json.loads((folder/'evaluation_results.json').read_bytes())
 # The exact retained pairing function operates on these already frozen reports.
 _,reports,labels,hallucination_pairs=ns['_compute_fairness_metrics'](rows,pairs,None)
 for model,rpairs in hallucination_pairs.items():
  diffs=[abs(fpr(a,g)-fpr(b,g)) for (a,b),g in zip(rpairs,labels[model],strict=True)]
  saved=summary['panels'][short]['models'][model]['primary_secondary_stats']['hallucination']
  sourceids={r['pair_source_id'] for r in rows if r['model']==model}
  results[model]={'saved_n_pairs':saved['n_pairs'],'recovered_code_n_pairs':len(diffs),'mean_absolute_difference':math.fsum(diffs)/len(diffs),'saved_mean_disparity':saved['mean_disparity'],'absolute_error':abs(math.fsum(diffs)/len(diffs)-saved['mean_disparity']),'input_source_count':len(sourceids),'input_sources_outside_core_count':len(sourceids-core),'denominator_matches':len(diffs)==saved['n_pairs']}
receipt={'record_type':'later_recovered_source_pairing_verification','source_commit':COMMIT,'source_commit_parent':'889358c42a52f3406005f0939e1f20ad698d7be4','source_commit_recorded_date':'2026-04-27T02:01:58-04:00','source_sha256':EXPECTED,'input_sha256':INPUT_HASHES,'method':'Exact AST-isolated _compute_fairness_metrics from recovered April27 source, original pure-Python refusal/lexical/hallucination modules, annotation category constants parsed without importing pandas; aggregate_metrics stub returns empty dictionary. No model, image generation, metric bootstrap or checkpoint execution. Computed means only to verify existing saved numbers.','models':results,'all_denominators_match':all(v['denominator_matches'] for v in results.values()),'maximum_absolute_error':max(v['absolute_error'] for v in results.values()),'remaining_limitation':'April27 Git metadata and its source bytes do not authenticate the git_dirty=true April20 execution; exact executed source binding remains missing.'}
receipt['verified']=len(results)==9 and receipt['all_denominators_match'] and receipt['maximum_absolute_error'] <= 1e-12 and all(v['input_source_count']==1000 and v['input_sources_outside_core_count']==0 for v in results.values())
encoded=json.dumps(receipt,indent=2)+'\n'
if args.output:args.output.write_text(encoded,encoding='utf-8')
print(encoded,end='')
raise SystemExit(0 if receipt['verified'] else 1)
