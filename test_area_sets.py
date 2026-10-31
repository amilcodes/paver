import unittest
from acquire_area_sets import traverse,distance_m


class RouteTests(unittest.TestCase):
    def fixture(self):
        return {str(i):dict(id=str(i),collection='A',links=([dict(rel='prev',id=str(i-1))] if i else [])+([dict(rel='next',id=str(i+1))] if i<6 else [])) for i in range(7)}

    def test_bounded_chronological_order(self):
        f=self.fixture();items,stops=traverse(f['3'],f.__getitem__,2)
        self.assertEqual([x['id'] for x in items],['1','2','3','4','5'])
        self.assertEqual(set(stops.values()),{'bounded window radius reached'})

    def test_complete_endpoint_preserved(self):
        f=self.fixture();items,stops=traverse(f['2'],f.__getitem__,9)
        self.assertEqual([x['id'] for x in items],[str(i) for i in range(7)])
        self.assertEqual(set(stops.values()),{'collection endpoint'})

    def test_cross_collection_and_cycle_rejected(self):
        f=self.fixture();f['4']['collection']='B'
        with self.assertRaises(ValueError):traverse(f['3'],f.__getitem__,2)
        f=self.fixture();f['4']['links']=[dict(rel='next',id='3')]
        with self.assertRaises(ValueError):traverse(f['3'],f.__getitem__,2)

    def test_distance_units(self):
        self.assertAlmostEqual(distance_m([0,0],[0,1]),111194.93,places=2)


if __name__=='__main__':unittest.main()
