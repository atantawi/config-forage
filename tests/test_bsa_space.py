from config_forage.bsa.space import Axis, DecisionSpace

SPACE6 = DecisionSpace(tuple(Axis(f"a{i}", f"axis {i}", tuple(f"a{i}o{j}" for j in range(n)))
                             for i, n in enumerate((4, 4, 3, 3, 2, 3))))


def test_shape():
    assert SPACE6.sizes == (4, 4, 3, 3, 2, 3)
    assert SPACE6.n_combinations() == 864


def test_brief_names_every_chosen_option():
    d = (0, 1, 2, 0, 1, 2)
    brief = SPACE6.render_brief(d)
    for axis, i in zip(SPACE6.axes, d):
        assert axis.name in brief and axis.options[i] in brief


def test_hint_lists_all_options():
    hint = SPACE6.render_hint()
    assert all(o in hint for a in SPACE6.axes for o in a.options)


def test_brief_and_labels_skip_unlabelled_axes():
    d = (0, 0, None, None, 0, None)
    brief = SPACE6.render_brief(d)
    assert SPACE6.axes[2].name not in brief and SPACE6.axes[1].options[0] in brief
    assert set(SPACE6.labels(d)) == {SPACE6.axes[m].name for m in (0, 1, 4)}


def test_json_round_trip():
    assert DecisionSpace.from_json(SPACE6.to_json()) == SPACE6
