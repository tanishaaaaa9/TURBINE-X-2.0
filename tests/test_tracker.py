from src.utils.tracker import load_state, check_outputs

def test_registry_has_unique_ids():
    ids=[t["task_id"] for t in load_state()["tasks"]]
    assert len(ids)==len(set(ids))

def test_missing_expected_output_is_detected():
    task={"expected_outputs":["this_path_must_not_exist_for_test.txt"]}
    assert check_outputs(task)
