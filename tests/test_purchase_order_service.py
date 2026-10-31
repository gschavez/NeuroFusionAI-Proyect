
import pytest

from services.purchase_order_service import PurchaseOrderService


@pytest.fixture
def purchase_order_service():
    return PurchaseOrderService(initial_budget=1000)


def test_approve_order_within_budget(purchase_order_service):
    purchase_order_service.create_order(
        order_id=1,
        amount=500
    )

    order = purchase_order_service.approve_order(1)

    assert order.status == "APROBADA"
    assert purchase_order_service.budget_remaining == 500


def test_reject_order_exceeding_budget(purchase_order_service):
    purchase_order_service.create_order(
        order_id=2,
        amount=1500
    )

    with pytest.raises(ValueError, match="presupuesto"):
        purchase_order_service.approve_order(2)

    assert purchase_order_service.budget_remaining == 1000


def test_cannot_approve_order_twice(purchase_order_service):
    purchase_order_service.create_order(
        order_id=3,
        amount=300
    )

    purchase_order_service.approve_order(3)

    with pytest.raises(ValueError, match="aprobada"):
        purchase_order_service.approve_order(3)

    assert purchase_order_service.budget_remaining == 700


def test_reject_nonexistent_order(purchase_order_service):
    with pytest.raises(ValueError, match="existe"):
        purchase_order_service.approve_order(999)
