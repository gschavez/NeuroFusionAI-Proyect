
from dataclasses import dataclass


@dataclass
class PurchaseOrder:
    order_id: int
    amount: float
    status: str = "PENDIENTE"


class PurchaseOrderService:

    def __init__(self, initial_budget: float):
        if initial_budget < 0:
            raise ValueError("El presupuesto no puede ser negativo")

        self._budget_remaining = initial_budget
        self._orders = {}

    @property
    def budget_remaining(self):
        return self._budget_remaining

    def create_order(self, order_id: int, amount: float) -> PurchaseOrder:
        if amount <= 0:
            raise ValueError("El monto de la orden debe ser mayor que cero")

        if order_id in self._orders:
            raise ValueError("La orden ya existe")

        order = PurchaseOrder(
            order_id=order_id,
            amount=amount
        )

        self._orders[order_id] = order

        return order

    def approve_order(self, order_id: int) -> PurchaseOrder:

        # Verify that the order exists
        if order_id not in self._orders:
            raise ValueError("La orden no existe")

        order = self._orders[order_id]

        # Prevent approving an order more than once
        if order.status == "APROBADA":
            raise ValueError("La orden ya está aprobada")

        # Verify that the available budget is sufficient
        if order.amount > self._budget_remaining:
            raise ValueError("La orden supera el presupuesto disponible")

        # Deduct the order amount from the budget
        self._budget_remaining -= order.amount

        # Update the order status
        order.status = "APROBADA"

        return order
