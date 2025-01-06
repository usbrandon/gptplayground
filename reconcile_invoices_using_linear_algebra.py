import pulp

# Example data
invoices = [1000, 2000, 3000]
payments = [6000]

# Create a PuLP Problem
prob = pulp.LpProblem("ReconciliationProblem", pulp.LpMinimize)

# Decision variables: x[i][j]
# (Amount of invoice i matched against payment j)
x = {}
for i in range(len(invoices)):
    for j in range(len(payments)):
        x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat=pulp.LpContinuous)

# Objective: minimize leftover (unmatched amounts)
# leftover_invoices = sum of (invoice amount - matched portion)
# leftover_payments = sum of (payment amount - matched portion)
leftover_invoices = []
leftover_payments = []
for i in range(len(invoices)):
    leftover_invoices.append(
        invoices[i] - pulp.lpSum([x[(i, j)] for j in range(len(payments))])
    )
for j in range(len(payments)):
    leftover_payments.append(
        payments[j] - pulp.lpSum([x[(i, j)] for i in range(len(invoices))])
    )

prob += pulp.lpSum(leftover_invoices) + pulp.lpSum(leftover_payments)

# Constraints
# 1) Each invoice cannot be overmatched
for i in range(len(invoices)):
    prob += pulp.lpSum([x[(i, j)] for j in range(len(payments))]) <= invoices[i]

# 2) Each payment cannot overpay
for j in range(len(payments)):
    prob += pulp.lpSum([x[(i, j)] for i in range(len(invoices))]) <= payments[j]

# Solve
prob.solve(pulp.PULP_CBC_CMD(msg=0))

# Results
for i in range(len(invoices)):
    for j in range(len(payments)):
        val = pulp.value(x[(i, j)])
        if val > 0:
            print(f"Invoice {i} matched with Payment {j}: {val}")

print(f"Objective (total leftover) = {pulp.value(prob.objective)}")
