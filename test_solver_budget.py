"""Checks for cached evidence isolation and budget-failure abstention."""
import json, tempfile, unittest
from pathlib import Path
import numpy as np
from run_solver_budget import cached_edges, gradient_partitions
from sequence_geometry import propose_edge, transform
from metric_plane import fit_metric_graph
from constrained_sequence import audit_poses
from dense_experiment import masked_inputs, array_hash
from benchmark import digest


class BudgetTests(unittest.TestCase):
    def test_cache_and_partitions_are_exact_and_tampering_fails(self):
        rng = np.random.default_rng(4); p = rng.uniform(30, 450, (120, 2)).astype(np.float32)
        q = (p + [5., 7.]).astype(np.float32); edge, info = propose_edge(0, 1, p, q)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pair.npz'; np.savez(path, p=p, q=q)
            record = dict(correspondences=[dict(i=0, j=1, path=str(path), sha256=digest(path), proposal=info)])
            edges, a = cached_edges(record); _, b = cached_edges(record)
            self.assertEqual(a, b)
            for k in ['p', 'q', 'vp', 'vq', 'H']: self.assertEqual(array_hash(edges[0][k]), array_hash(edge[k]))
            path.write_bytes(path.read_bytes()+b'tampered')
            with self.assertRaises(Exception): cached_edges(record)

    def test_budget_exhaustion_remains_abstention(self):
        rng = np.random.default_rng(3); p = rng.uniform(20, 480, (120, 2)).astype(np.float32)
        # Noisy projective observations require refinement of the physical camera model.
        H = np.array([[1.02, .03, 7], [-.01, 1.01, 35], [1e-4, -2e-4, 1.]])
        q = (transform(H, p)+rng.normal(0, .4, p.shape)).astype(np.float32)
        edge, _ = propose_edge(0, 1, p, q); self.assertIsNotNone(edge)
        poses, info, active = fit_metric_graph(2, 1, [edge], 512, 150., max_nfev=1)
        self.assertFalse(info['optimizer_success'])
        images = [np.zeros((512, 512, 3), np.uint8) for _ in range(2)]; masks = [np.ones((512, 512), bool) for _ in range(2)]
        regs = audit_poses(images, masks, poses, info, active, 1)
        self.assertFalse(regs[0].diagnostics['accepted']); self.assertFalse(regs[0].support.any())

    def test_hidden_rgb_cannot_change_registration_inputs(self):
        rng = np.random.default_rng(8); images = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        allowed = [np.ones((64, 64), bool) for _ in images]; allowed[0][:8] = False
        hidden = np.zeros((64, 64), bool); hidden[20:40, 20:40] = True
        clean, a = masked_inputs(images, allowed, hidden, 1)
        altered = [x.copy() for x in images]; altered[1][hidden] ^= 255; altered[0][~allowed[0]] ^= 255
        other, b = masked_inputs(altered, allowed, hidden, 1)
        self.assertEqual([array_hash(x) for x in clean], [array_hash(x) for x in other])
        self.assertEqual([array_hash(x) for x in a], [array_hash(x) for x in b])


if __name__ == '__main__': unittest.main()
