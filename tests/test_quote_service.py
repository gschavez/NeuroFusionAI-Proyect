
import pytest

from services.quote_service import QuoteService


@pytest.fixture
def quote_service():
    return QuoteService()


@pytest.mark.parametrize("amount", [100, 500, 1000])
def test_create_quote_with_valid_amounts(quote_service, amount):
    quote = quote_service.create_quote(
        request_id=1,
        supplier_id=10,
        amount=amount,
        request_status="ABIERTA"
    )

    assert quote.amount == amount
    assert quote.request_id == 1
    assert quote.supplier_id == 10


def test_reject_quote_for_closed_request(quote_service):
    with pytest.raises(ValueError, match="abierta"):
        quote_service.create_quote(
            request_id=1,
            supplier_id=10,
            amount=500,
            request_status="CERRADA"
        )


@pytest.mark.parametrize("amount", [0, -100])
def test_reject_invalid_amount(quote_service, amount):
    with pytest.raises(ValueError, match="mayor que cero"):
        quote_service.create_quote(
            request_id=1,
            supplier_id=10,
            amount=amount,
            request_status="ABIERTA"
        )


def test_reject_duplicate_quote(quote_service):
    quote_service.create_quote(
        request_id=1,
        supplier_id=10,
        amount=500,
        request_status="ABIERTA"
    )

    with pytest.raises(ValueError, match="duplicada"):
        quote_service.create_quote(
            request_id=1,
            supplier_id=10,
            amount=700,
            request_status="ABIERTA"
        )
