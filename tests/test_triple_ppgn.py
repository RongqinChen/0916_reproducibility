"""Legacy backbone parity, feature alignment, cached gradients and real CLI resume."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from brec.determinism import prepare_environment
prepare_environment()
import networkx as nx
import numpy as np
import torch

from rebuttal_experiments.common import sha256
from rebuttal_experiments.encodings import feature_matrix
from rebuttal_experiments.legacy_triple_ppgn import FixedInput, RealizationPPGN, _Contracted3
from rebuttal_experiments.triple_ppgn import validate_config
from rebuttal_experiments.triple_ppgn_inputs import input_array, q1_joint
from rebuttal_experiments.triple_ppgn_training import embedding_cached_step

ROOT = Path(__file__).resolve().parents[1]


class TriplePPGNTests(unittest.TestCase):
    def test_feature_prefix_matches_borrowed_input_and_joint_oracle(self):
        for graph in (nx.path_graph(4), nx.star_graph(4), nx.cycle_graph(4)):
            adjacency = nx.to_numpy_array(graph, dtype='int64')
            legacy = FixedInput(2, cache_bytes=0, tensor_cache_bytes=0).array(adjacency, 3)
            variants = {name: input_array(adjacency, name, 2) for name in ('original','baseline','joint')}
            for value in variants.values():
                np.testing.assert_array_equal(value[:25].transpose(1,2,3,0), legacy)
                self.assertEqual(value.shape, (52, len(graph), len(graph), len(graph)))
            self.assertTrue(np.all(variants['original'][25:] == 0))
            np.testing.assert_array_equal(variants['baseline'][25:44], variants['baseline'][:19])
            np.testing.assert_array_equal(variants['baseline'][:44], variants['joint'][:44])
            self.assertTrue(np.all(variants['baseline'][44:] == 0))
            expected = feature_matrix(adjacency, 1, 3, 'joint', mode='float64')[:, -8:]
            np.testing.assert_allclose(variants['joint'][44:].transpose(1,2,3,0).reshape(-1,8),
                                       expected, atol=2e-7, rtol=2e-6)

    def test_contraction_forward_and_backward(self):
        values = [torch.randn(1,2,2,2,2,dtype=torch.double,requires_grad=True) for _ in range(3)]
        expected = torch.einsum('btvwp,butwp,buvtp->buvwp', *values)
        torch.testing.assert_close(_Contracted3.apply(*values), expected)
        self.assertTrue(torch.autograd.gradcheck(_Contracted3.apply, tuple(values)))

    def test_embedding_cache_matches_full_batch_optimizer_step(self):
        torch.manual_seed(41)
        model = RealizationPPGN(3, 1, 3, 2, 3, input_width=5).double()
        baseline = copy.deepcopy(model)
        graphs = torch.randn(4,5,3,3,3,dtype=torch.double)
        cached_optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=.01)
        embeddings = baseline(graphs)
        loss = torch.nn.functional.cosine_embedding_loss(embeddings[::2], embeddings[1::2],
                                                         -torch.ones(2,dtype=torch.double))
        loss.backward()
        baseline_optimizer.step()
        actual = embedding_cached_step(model, list(range(4)), lambda i: graphs[i:i+1],
                                        cached_optimizer, clip=None)
        self.assertAlmostEqual(actual, float(loss.detach()), places=12)
        for first, second in zip(model.parameters(), baseline.parameters()):
            torch.testing.assert_close(first, second, atol=2e-11, rtol=2e-10)

    def test_checkpoint_and_relabeling_preserve_model_output(self):
        torch.manual_seed(7)
        model = RealizationPPGN(3, 1, 3, 2, 3, input_width=5).double()
        checkpointed = copy.deepcopy(model)
        checkpointed.grad_checkpoint = True
        value = torch.randn(1,5,3,3,3,dtype=torch.double)
        original = model(value)
        original.square().sum().backward()
        changed = checkpointed(value)
        changed.square().sum().backward()
        torch.testing.assert_close(original, changed)
        for a,b in zip(model.parameters(),checkpointed.parameters()):
            torch.testing.assert_close(a.grad,b.grad)
        permutation = [2,0,1]
        with torch.no_grad():
            transformed = model(value[:,:,permutation][:,:,:,permutation][:,:,:,:,permutation])
        torch.testing.assert_close(original, transformed, atol=2e-9, rtol=2e-9)

    def test_matching_parameter_counts(self):
        model = RealizationPPGN(3,8,16,4,16,input_width=88)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()),18656)

    def test_configuration_rejects_unfrozen_paths(self):
        config = json.loads((ROOT/'configs/rebuttal/triple_ppgn.json').read_text())
        validate_config(config)
        for key,value in [('physical_batch',2),('dtype','bfloat16'),('triple_q',2),('input_width',61),
                          ('effective_batch',3),('seconds_per_condition',300)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_config({**config,key:value})

    def test_real_worker_cli_and_resume_preserves_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = nx.to_graph6_bytes(nx.path_graph(3),header=False).strip()
            second = nx.to_graph6_bytes(nx.cycle_graph(3),header=False).strip()
            data = np.full(51200,first,dtype='S16')
            for i in range(32):
                data[2*i+1] = second
            source = root/'brec.npy'
            np.save(source,data,allow_pickle=False)
            config = json.loads((ROOT/'configs/rebuttal/triple_ppgn.json').read_text())
            config.update(expected_brec_sha256=sha256(source), seed_q=2,input_width=52,
                          hidden_dim=2,layers=1,output_dim=2,epochs=1,seeds=[11],
                          input_cache_gib=.01,memory_gib=6,reserve_gib=0)
            config_file = root/'config.json'
            config_file.write_text(json.dumps(config))
            command = [sys.executable,'-m','rebuttal_experiments.triple_ppgn',
                       '--config',str(config_file),'--brec-file',str(source),
                       '--pair-ids','0','--devices','cpu','--out',str(root/'run')]
            env = dict(os.environ,PYTHONHASHSEED='0')
            result = subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr+
                '\n'+''.join(p.read_text() for p in (root/'run/logs').glob('*.log')))
            files = list((root/'run/records').glob('*.json'))
            self.assertEqual(len(files),3)
            self.assertTrue(all(json.loads(p.read_text())['status']=='complete' for p in files))
            before = {str(p.relative_to(root/'run')):p.read_bytes()
                      for p in (root/'run/artifacts').glob('*')}
            result = subprocess.run(command+['--resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(before,{str(p.relative_to(root/'run')):p.read_bytes()
                                     for p in (root/'run/artifacts').glob('*')})
            self.assertEqual(json.loads((root/'run/progress.json').read_text())['status'],'finished')


if __name__ == '__main__':
    unittest.main()
