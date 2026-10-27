"""D01-D14 dataset tests, matching Table 44 of Chapter 3.

Run with:  python -m unittest -v test_datasets
"""
import unittest
import curriculum as C
from engine import (analyze, band, is_valid_attempt, targets,
                    rank_resources, prerequisite_influence, WEIGHTS)

KEY = C.key_letters()
KEY_B = C.set_key("B")
KEY_C = C.set_key("C")
CATS = C.categories()
SKILLS = C.skills()
MODS = C.MODULES
EDGES = C.EDGES


def run(answers, key=KEY, dmap=None):
    return analyze(answers, key, CATS, SKILLS, EDGES, MODS, distractor_map=dmap)


def _wrong(i):
    return next(x for x in "ABCD" if x != KEY[i])


class DatasetTests(unittest.TestCase):
    def test_D01_all_correct(self):
        res = run(list(KEY))
        self.assertAlmostEqual(res["course"], 100.0, places=6)
        self.assertTrue(all(abs(s["P"] - 100) < 1e-6 for s in res["sk"]))
        self.assertEqual(res["prereq_causes"], [])

    def test_D02_all_incorrect_no_tags_means_all_unclassified(self):
        res = run([_wrong(i) for i in range(100)])
        self.assertEqual(res["course"], 0)
        self.assertEqual(res["err"]["U"], 100)

    def test_D02_all_incorrect_with_tags_means_all_classified(self):
        dmap = {(i, _wrong(i)): CATS[i] for i in range(100)}
        res = run([_wrong(i) for i in range(100)], dmap=dmap)
        self.assertEqual(res["course"], 0)
        self.assertEqual(res["err"]["U"], 0)

    def test_D04_late_skill_failure_no_false_cause(self):
        answers = list(KEY)
        for i in range(90, 100):
            answers[i] = _wrong(i)
        res = run(answers)
        prof = [s["P"] for s in res["sk"]]
        for k in range(len(SKILLS)):
            if SKILLS[k][1] != 9 or prof[k] >= 60:
                continue
            for prereq, h, inf in prerequisite_influence(k, prof, EDGES):
                # Only same-module prerequisites are legitimate in this scenario.
                self.assertEqual(
                    SKILLS[prereq][1], 9,
                    f"Skill {k} in module 9 flagged module-{SKILLS[prereq][1]} "
                    f"skill {prereq} as a cause"
                )

    def test_D05_not_assessed_skills_are_still_in_module_mean(self):
        # Module index 5 = "Variability"; its items are positions 50-59.
        answers = ["-"] * 100
        for i in range(50, 60):
            answers[i] = KEY[i]
        res = run(answers)
        mod5 = next(m for m in res["mods"] if m["name"] == MODS[5])
        self.assertAlmostEqual(mod5["P"], 100.0, places=6)

    def test_D06_developing_boundary(self):
        self.assertEqual(band(60), "Developing")
        self.assertEqual(band(79.99), "Developing")
        self.assertEqual(band(80), "Mastered")

    def test_D08_exactly_50_pct_is_valid(self):
        self.assertTrue(is_valid_attempt(50, 100))

    def test_D09_below_50_pct_is_invalid(self):
        self.assertFalse(is_valid_attempt(49, 100))

    def test_prereq_walk_uses_H_and_gamma(self):
        # Blank Mean (11) AND SD itself (15) so SD is below 60 and can be a target.
        answers = list(KEY)
        for i in SKILLS[11][3]:
            answers[i] = "-"
        for i in SKILLS[15][3]:
            answers[i] = "-"
        res = run(answers)
        causes_for_sd = [c for c in res["prereq_causes"]
                         if c["target_skill_id"] == 15]
        self.assertTrue(causes_for_sd)
        top = causes_for_sd[0]
        self.assertEqual(top["candidate_prerequisite_skill_id"], 11)
        self.assertEqual(top["h"], 1)
        self.assertAlmostEqual(top["influence"], 80.0, places=3)


if __name__ == "__main__":
    unittest.main()