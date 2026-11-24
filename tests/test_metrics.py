from src.data.polyvore import FitbQuestion
from src.eval.metrics import compat_auc, evaluate, fitb_accuracy, outfit_score

# Toy world: items with the same first letter are "compatible".
def pair_score(a: str, b: str) -> float:
    return 1.0 if a[0] == b[0] else 0.0


def test_outfit_score_is_mean_of_pairs():
    assert outfit_score(["a1", "a2", "a3"], pair_score) == 1.0
    assert outfit_score(["a1", "a2", "b1"], pair_score) == 1 / 3
    assert outfit_score(["a1"], pair_score) == 0.0


def test_auc_perfect_and_inverted():
    assert compat_auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert compat_auc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0


def test_fitb_accuracy():
    qs = [
        FitbQuestion(["a1", "a2"], ["b1", "a3", "c1", "d1"], answer=1),   # model gets it right
        FitbQuestion(["a1", "a2"], ["a3", "b1", "c1", "d1"], answer=1),   # model picks a3, wrong
    ]
    assert fitb_accuracy(qs, pair_score) == 0.5


def test_fitb_ties_are_split_not_given_to_first_slot():
    # all four candidates score identically -> 1/4 credit regardless of answer position
    for ans in range(4):
        q = FitbQuestion(["a1"], ["b1", "c1", "d1", "e1"], answer=ans)
        assert fitb_accuracy([q], pair_score) == 0.25


def test_fitb_duplicate_answer_item_counts_as_correct():
    # the answer item "b1" is listed twice; the model ties them at the top -> full credit
    q = FitbQuestion(["b0"], ["c1", "b1", "d1", "b1"], answer=1)
    assert fitb_accuracy([q], pair_score) == 1.0


def test_evaluate_end_to_end():
    compat = [(["a1", "a2", "a3"], 1), (["a1", "b1", "c1"], 0)]
    fitb = [FitbQuestion(["a1"], ["a2", "b1", "c1", "d1"], answer=0)]
    m = evaluate(pair_score, compat, fitb)
    assert m == {"compat_auc": 1.0, "fitb_acc": 1.0}
