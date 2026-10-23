import unittest

from corpus_rule import clean, eligible_frame, normalise, sample_candidates


class CorpusSamplingTests(unittest.TestCase):
    def test_clean_normalise_and_benchmark_exclusion(self):
        text = 'Company discusses its quarterly financial results and plans at the meeting'
        benchmark = [{'id': 'benchmark', 'text': text}]
        rows = [
            ('benchmark', text + ' tomorrow', 'X', 's', 'news', '2026-01-01'),
            ('copy', text.upper() + '!!! https://example.com/x', 'X', 's', 'news', '2026-01-01'),
            ('a', 'Investors discuss the dividend calendar and record dates at the annual meeting', 'Y', 's', 'news', '2026-01-01'),
            ('b', 'Investors discuss the dividend calendar and record dates at the annual meeting!', 'Y', 's', 'news', '2026-01-01'),
            ('c', 'too short', 'Z', 's', 'news', '2026-01-01'),
        ]
        frame, _ = eligible_frame(rows, benchmark, {'dividend', 'record'})
        self.assertEqual([r['id'] for r in frame], ['a'])
        self.assertEqual(frame[0]['lexicon_hits'], ['dividend', 'record'])
        self.assertEqual(normalise(clean(rows[1][1])), normalise(text))

    def test_random_draw_is_unrestricted_and_repeatable(self):
        frame = [{'id': str(i), 'lexicon_hits': ['profit'] if i % 2 else []} for i in range(100)]
        first = sample_candidates(frame, seed=41, n_random=20, n_loaded=30)
        self.assertEqual(first, sample_candidates(list(reversed(frame)), seed=41, n_random=20, n_loaded=30))
        self.assertEqual(len({r['id'] for r in first}), 50)
        random = [r for r in first if r['sampling_group'] == 'random']
        loaded = [r for r in first if r['sampling_group'] == 'lexicon_hit']
        self.assertEqual(len(random), 20)
        self.assertTrue(any(r['lexicon_hits'] for r in random))
        self.assertTrue(any(not r['lexicon_hits'] for r in random))
        self.assertTrue(all(r['lexicon_hits'] for r in loaded))
        with self.assertRaises(ValueError):
            sample_candidates(frame, seed=41, n_random=20, n_loaded=100)


if __name__ == '__main__':
    unittest.main()
