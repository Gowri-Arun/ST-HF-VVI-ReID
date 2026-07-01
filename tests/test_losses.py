import torch

from losses.id_loss import IDLoss
from losses.triplet_loss import TripletLoss
from losses.build_loss import ReIDLossBuilder, build_loss


def test_id_loss_returns_finite_scalar():
    loss_fn = IDLoss()
    logits = torch.randn(4, 10)
    pids = torch.randint(0, 10, (4,))
    loss = loss_fn(logits, pids)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_triplet_loss_returns_finite_scalar():
    loss_fn = TripletLoss()
    features = torch.randn(4, 2048)
    pids = torch.tensor([0, 0, 1, 1])
    loss = loss_fn(features, pids)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_combined_loss_keys():
    criterion = ReIDLossBuilder()
    outputs = {
        "features": torch.randn(4, 2048),
        "logits": torch.randn(4, 10),
        "int_features": None,
        "int_logits": None,
        "extra": {"model_type": "baseline"},
    }
    pids = torch.tensor([0, 0, 1, 1])
    losses = criterion(outputs, pids)
    expected_keys = {"loss_total", "loss_id", "loss_tri", "loss_id_int", "loss_tri_int"}
    assert set(losses.keys()) == expected_keys


def test_baseline_int_losses_zero():
    criterion = ReIDLossBuilder()
    outputs = {
        "features": torch.randn(4, 2048),
        "logits": torch.randn(4, 10),
        "int_features": None,
        "int_logits": None,
        "extra": {"model_type": "baseline"},
    }
    pids = torch.tensor([0, 0, 1, 1])
    losses = criterion(outputs, pids)
    assert losses["loss_id_int"].item() == 0.0
    assert losses["loss_tri_int"].item() == 0.0


def test_total_loss_finite():
    criterion = ReIDLossBuilder()
    outputs = {
        "features": torch.randn(4, 2048),
        "logits": torch.randn(4, 10),
        "int_features": None,
        "int_logits": None,
        "extra": {"model_type": "baseline"},
    }
    pids = torch.tensor([0, 0, 1, 1])
    losses = criterion(outputs, pids)
    assert torch.isfinite(losses["loss_total"])


def test_sthf_intermediate_losses_active():
    criterion = ReIDLossBuilder()
    outputs = {
        "features": torch.randn(4, 2048),
        "logits": torch.randn(4, 10),
        "int_features": torch.randn(4, 2048),
        "int_logits": torch.randn(4, 10),
        "extra": {"model_type": "sthf_fixed"},
    }
    pids = torch.tensor([0, 0, 1, 1])
    losses = criterion(outputs, pids)
    assert losses["loss_id_int"].item() > 0.0
    assert losses["loss_tri_int"].item() > 0.0


def test_total_loss_is_weighted_sum():
    criterion = ReIDLossBuilder(
        lambda_id=1.0, lambda_tri=1.0,
        lambda_int_id=2.0, lambda_int_tri=3.0,
    )
    outputs = {
        "features": torch.randn(4, 2048),
        "logits": torch.randn(4, 10),
        "int_features": torch.randn(4, 2048),
        "int_logits": torch.randn(4, 10),
        "extra": {"model_type": "sthf_fixed"},
    }
    pids = torch.tensor([0, 0, 1, 1])
    losses = criterion(outputs, pids)
    expected = (
        1.0 * losses["loss_id"].item()
        + 1.0 * losses["loss_tri"].item()
        + 2.0 * losses["loss_id_int"].item()
        + 3.0 * losses["loss_tri_int"].item()
    )
    assert abs(losses["loss_total"].item() - expected) < 1e-5


def test_baseline_total_equals_main_only():
    criterion = ReIDLossBuilder(
        lambda_id=1.0, lambda_tri=1.0,
        lambda_int_id=1.0, lambda_int_tri=1.0,
    )
    outputs = {
        "features": torch.randn(4, 2048),
        "logits": torch.randn(4, 10),
        "int_features": None,
        "int_logits": None,
        "extra": {"model_type": "baseline"},
    }
    pids = torch.tensor([0, 0, 1, 1])
    losses = criterion(outputs, pids)
    expected = losses["loss_id"].item() + losses["loss_tri"].item()
    assert abs(losses["loss_total"].item() - expected) < 1e-5


def test_id_loss_no_softmax_before_ce():
    loss_fn = IDLoss()
    logits = torch.full((4, 3), 10.0)
    pids = torch.tensor([0, 1, 2, 0])
    loss = loss_fn(logits, pids)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_triplet_loss_no_features_as_logits():
    loss_fn = TripletLoss(margin=0.3)
    features = torch.randn(8, 2048)
    pids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(features, pids)
    assert torch.isfinite(loss)
    assert loss.ndim == 0
