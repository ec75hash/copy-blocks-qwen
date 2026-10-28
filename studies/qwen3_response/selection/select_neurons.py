"""Freeze neuron IDs and random controls before collecting activation data."""
from pathlib import Path
import hashlib
import json
import random

HERE = Path(__file__).resolve().parent
SEED = 20261026
rng = random.Random(SEED)
segments = [('A', 0, 8192), ('B', 8192, 17408), ('C', 17408, 25600)]
targets = []
for layer in range(8, 32):
    for label, start, end in segments:
        for neuron in sorted(rng.sample(range(start, end), 4)):
            targets.append({'layer_a': layer, 'layer_b': layer + 24,
                            'neuron': neuron, 'segment': label})

# Select all targets first so that changing control-bank size cannot change them.
for target in targets:
    i = target['neuron']
    controls = rng.sample(range(25599), 100)
    target['random_controls_at_layer_b'] = sorted(j if j < i else j + 1 for j in controls)

assert len(targets) == 288
assert len({(t['layer_a'], t['neuron']) for t in targets}) == 288
for t in targets:
    assert t['layer_b'] - t['layer_a'] == 24
    assert len(set(t['random_controls_at_layer_b'])) == 100
    assert t['neuron'] not in t['random_controls_at_layer_b']
    assert all(0 <= j < 25600 for j in t['random_controls_at_layer_b'])

record = {'checkpoint': 'Qwen/Qwen3-32B',
          'revision': '9216db5781bf21249d130ec9da846c4624c16137',
          'seed': SEED, 'indexing': 'original zero-based decoder layer and neuron indices',
          'matched_controls': 'pending discovery-only selection; not selected from test responses',
          'targets': targets}
data = (json.dumps(record, indent=2) + '\n').encode()
out = HERE / 'selection.json'
if out.exists() and out.read_bytes() != data:
    raise SystemExit('Existing selection differs; preserve it and record a protocol amendment.')
if not out.exists():
    out.write_bytes(data)
print(f'{len(targets)} neuron pairs; 100 random controls per pair')
print(f'SHA-256 {hashlib.sha256(data).hexdigest()}')
