import unittest
import numpy as np
from sampling_metrics import partitions
from audit_sampling import expected_partitions
from validate_dense_run_v2 import compare


class PartitionTests(unittest.TestCase):
    def fixture(self):
        mask=np.ones((2,2),bool);truth=np.full((2,2,3),50,np.uint8);values=[]
        for counts,pixels in [(np.array([[1,1],[0,0]],np.uint8),np.array([[53]*3,[62]*3,[0]*3,[0]*3],np.uint8)),(np.array([[1,0],[1,0]],np.uint8),np.array([[56]*3,[0]*3,[80]*3,[0]*3],np.uint8))]:
            values.append(dict(mask=mask,count=counts,disagreement=np.zeros((2,2),np.float32),prediction_on_mask=pixels,absolute_rgb_error_sum_on_mask=np.abs(pixels.astype(np.int16)-truth[mask]).sum(1).astype(np.uint16)))
        return *values,truth,dict(candidate=dict(min_views=1,max_disagreement=18))

    def test_common_gained_lost_exact_errors(self):
        new,old,truth,policies=self.fixture();m=partitions(new,old,policies);compare(m,expected_partitions(new,old,truth,policies));v=m['candidate']
        self.assertEqual(v['common']['new']['mae_255'],3);self.assertEqual(v['common']['control']['mae_255'],6)
        self.assertEqual(v['gained']['new']['mae_255'],12);self.assertEqual(v['lost']['control']['mae_255'],30)
        self.assertEqual(v['lost']['control']['over_20_fraction'],1);self.assertEqual(v['gained']['fraction_of_mask'],.25)

    def test_natural_has_support_counts_only(self):
        new,old,truth,policies=self.fixture();m=partitions(new,old,policies,True);compare(m,expected_partitions(new,old,None,policies))
        self.assertEqual(set(m['candidate']['common']),{'pixels','fraction_of_mask'})

    def test_different_masks_and_changed_metrics_rejected(self):
        new,old,truth,policies=self.fixture();m=partitions(new,old,policies);m['candidate']['gained']['pixels']=4
        with self.assertRaises(ValueError):compare(m,expected_partitions(new,old,truth,policies))
        old['mask']=old['mask'].copy();old['mask'][0,0]=False
        with self.assertRaises(ValueError):partitions(new,old,policies)


if __name__=='__main__':unittest.main()
