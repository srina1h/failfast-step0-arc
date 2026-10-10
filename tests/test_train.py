from failfast_arc.train import binary_log_loss


def test_binary_log_loss_accepts_a_one_class_diagnostic_subset():
    value = binary_log_loss([0, 0], [0.2, 0.3])
    assert 0.0 < value < 1.0
