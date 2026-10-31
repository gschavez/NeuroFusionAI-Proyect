
from dataclasses import dataclass


@dataclass
class Quote:
    request_id: int
    supplier_id: int
    amount: float


class QuoteService:

    def __init__(self):
        self._quotes = []

    def create_quote(
        self,
        request_id: int,
        supplier_id: int,
        amount: float,
        request_status: str
    ) -> Quote:

        # Rule 1: The request must be open
        if str(request_status).upper() != "ABIERTA":
            raise ValueError("La solicitud debe estar abierta")

        # Rule 2: The quote amount must be greater than zero
        if amount <= 0:
            raise ValueError("El monto debe ser mayor que cero")

        # Rule 3: A supplier cannot submit duplicate quotes
        for quote in self._quotes:
            if (
                quote.request_id == request_id
                and quote.supplier_id == supplier_id
            ):
                raise ValueError("La cotización está duplicada")

        # Create and save the quote
        new_quote = Quote(
            request_id=request_id,
            supplier_id=supplier_id,
            amount=amount
        )

        self._quotes.append(new_quote)

        return new_quote
