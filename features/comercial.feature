Feature: Quote and purchase order management

Scenario Outline: Register a quote with a valid amount
Given an open request with identifier 1 exists
When I register a quote of <amount> for supplier 10
Then the quote is registered with amount <amount>

Examples:
| amount |
| 100 |
| 500 |
| 1000 |

Scenario: Approve an order within budget
Given an available budget of 1000 exists
And a purchase order with identifier 20 for an amount of 500 exists
When I approve purchase order 20
Then the purchase order is approved
And the remaining budget is 500