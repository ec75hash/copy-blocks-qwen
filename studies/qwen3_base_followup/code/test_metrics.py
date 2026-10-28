import unittest
import numpy as np
from metrics import sampled_rows, mapped_pairs, row_stats, cross_cosines


class MetricsTests(unittest.TestCase):
    def test_known_vectors_and_different_aggregation(self):
        a = np.array([[3., 0.], [0., 1.]])
        b = np.array([[3., 0.], [0., -1.]])
        stats, rows = row_stats(a, b)
        np.testing.assert_allclose(rows, [1, -1])
        self.assertEqual(stats["row_mean"], 0)
        self.assertAlmostEqual(stats["flat_cosine"], .8)
        np.testing.assert_allclose(cross_cosines(a, b), [[1, 0], [0, -1]])

    def test_zero_rows_remain_missing(self):
        stats, rows = row_stats([[0, 0], [1, 0]], [[1, 0], [1, 0]])
        self.assertTrue(np.isnan(rows[0]))
        self.assertEqual(stats["valid_rows"], 1)
        self.assertEqual(stats["row_mean"], 1)

    def test_fixed_proportional_contiguous_sampling(self):
        a = sampled_rows()
        np.testing.assert_array_equal(a, sampled_rows())
        self.assertEqual(len(set(a)), 544)
        self.assertEqual(sum(a < 8192), 256)
        self.assertEqual(sum(a >= 8192), 288)
        self.assertTrue(np.all((a >= 0) & (a < 17408)))
        self.assertTrue(all(np.array_equal(chunk, np.arange(chunk[0], chunk[0]+32))
                            for chunk in a.reshape(-1, 32)))

    def test_pair_map_includes_both_positions(self):
        pairs = mapped_pairs()
        self.assertEqual(len(pairs), 64)
        self.assertEqual(len({(s,t) for s,t,_ in pairs}), 64)
        self.assertEqual(sum(8 <= s <= 31 and g == "plus24" for s,t,g in pairs),24)
        self.assertIn((8,8,"same_index"),pairs)
        self.assertIn((8,32,"plus24"),pairs)


if __name__ == "__main__":
    unittest.main()
