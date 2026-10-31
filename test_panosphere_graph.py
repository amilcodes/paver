import unittest
from panosphere_graph import build_nodes_edges


class CandidateGraphTests(unittest.TestCase):
    def area(self):
        return {'a':dict(split='train',frames=[dict(id=str(i),index=i,collection='c',coordinates=[i*.00004,45],capture_time=f'2026-01-01T00:00:0{i}+00:00',azimuth_degrees=i*180,image_path=f'{i}.jpg',sample_sha256='x') for i in range(3)])}
    def test_links_are_directed_and_panoramas_ignore_heading_cut(self):
        nodes,edges=build_nodes_edges(self.area());self.assertEqual(len(nodes),3);self.assertEqual(len(edges),6)
        self.assertTrue(any(e['heading_difference_degrees']==180 for e in edges))
        self.assertTrue(all(e['status']=='candidate_not_visually_verified' and e['target']!=e['donor'] for e in edges))
    def test_distance_and_collection_gates(self):
        a=self.area();a['a']['frames'][1]['coordinates']=[2,45];a['a']['frames'][2]['collection']='other'
        self.assertEqual(build_nodes_edges(a)[1],[])
    def test_test_split_rejected(self):
        a=self.area();a['a']['split']='test'
        with self.assertRaises(ValueError):build_nodes_edges(a)


if __name__=='__main__':unittest.main()
