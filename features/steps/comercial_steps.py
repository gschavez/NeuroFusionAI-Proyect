
from behave import given, when, then

from services.quote_service import QuoteService
from services.purchase_order_service import PurchaseOrderService


# ==========================================
# QUOTE SERVICE STEPS
# ==========================================

@given('an open request with identifier {request_id:d} exists')
def step_open_request(context, request_id):
    context.quote_service = QuoteService()
    context.request_id = request_id
    context.request_status = "ABIERTA"


@when('I register a quote of {amount:d} for supplier {supplier_id:d}')
def step_register_quote(context, amount, supplier_id):
    context.quote = context.quote_service.create_quote(
        request_id=context.request_id,
        supplier_id=supplier_id,
        amount=amount,
        request_status=context.request_status
    )


@then('the quote is registered with amount {amount:d}')
def step_quote_registered(context, amount):
    assert context.quote.amount == amount
    assert context.quote.request_id == context.request_id


# ==========================================
# PURCHASE ORDER SERVICE STEPS
# ==========================================

@given('an available budget of {budget:d} exists')
def step_available_budget(context, budget):
    context.purchase_order_service = PurchaseOrderService(
        initial_budget=budget
    )


@given('a purchase order with identifier {order_id:d} for an amount of {amount:d} exists')
def step_purchase_order_exists(context, order_id, amount):
    context.order = context.purchase_order_service.create_order(
        order_id=order_id,
        amount=amount
    )


@when('I approve purchase order {order_id:d}')
def step_approve_purchase_order(context, order_id):
    context.approved_order = (
        context.purchase_order_service.approve_order(order_id)
    )


@then('the purchase order is approved')
def step_purchase_order_approved(context):
    assert context.approved_order.status == "APROBADA"


@then('the remaining budget is {remaining:d}')
def step_remaining_budget(context, remaining):
    assert context.purchase_order_service.budget_remaining == remaining
