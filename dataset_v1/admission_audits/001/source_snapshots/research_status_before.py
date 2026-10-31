"""One compact entry point for the recurring research loop and manual status."""
import json,shutil,sqlite3,subprocess
from pathlib import Path
from resource_guard import admission
ROOT=Path(__file__).resolve().parent
state=json.loads((ROOT/'autoresearch/state.json').read_text())
processes=subprocess.check_output(['ps','-ax','-o','pid=,etime=,command='],text=True)
names=['run_dense.py','run_annotated.py','run_metric.py','run_constrained.py','run_sequence.py','run_trajectory.py','run_long_sequence.py','dataset.py','audit_dataset.py']
running=[line.strip() for line in processes.splitlines() if 'python' in line.lower() and 'outputs/panorama-repair/' in line and any('outputs/panorama-repair/'+n in line for n in names) and ' -c ' not in line and ' /bin/' not in line]
# A shell command displaying this status may include literal filenames; only
# actual Python process command prefixes are counted by the argv position.
running=[line for line in running if line.split(maxsplit=2)[2].split()[0].endswith(('python','python3','Python')) or '/python' in line.split(maxsplit=2)[2].split()[0].lower()]
db=sqlite3.connect(ROOT/'dataset_v1/catalog.sqlite')
counts={k:db.execute(q).fetchone()[0] for k,q in {'records':'SELECT count(*) FROM pictures','eligible_collections':'SELECT count(DISTINCT collection) FROM pictures WHERE eligible=1','complete_cached_images':"SELECT count(*) FROM downloads WHERE status='complete'"}.items()}
limits=json.loads((ROOT/'autoresearch/policy.json').read_text())['resource_limits']
resources=admission(ROOT,running,minimum_free_gib=limits['minimum_free_disk_gib'],cache_cap_gib=limits['max_additional_image_cache_gib'],reserve_bytes=512*1024**2)
print(json.dumps(dict(state=state,running_jobs=running,**resources,dataset=counts),indent=2))
