import numpy as np

def initialize_simplex_table(c, A, b):
    m, n = A.shape
    table = np.zeros((m+1, m+n+1))
    table[0, 1:n+1] = -c
    table[1:, 0] = b
    table[1:, 1:n+1] = A
    table[1:, n+1:] = np.eye(m)
    return table

def pivot_selection(table):
    m, n = table.shape
    pivot_col = np.argmin(table[0, 1:n-1]) + 1  # 最も負の係数を持つ列を選択
    ratios = []
    for i in range(1, m):
        if table[i, pivot_col] > 0:
            ratios.append(table[i, 0] / table[i, pivot_col])
        else:
            ratios.append(np.inf)
    pivot_row = np.argmin(ratios) + 1
    return pivot_row, pivot_col

def pivot_operation(table, pivot_row, pivot_col):
    m, n = table.shape
    pivot_element = table[pivot_row, pivot_col]
    table[pivot_row, :] /= pivot_element
    for i in range(m):
        if i != pivot_row:
            multiplier = table[i, pivot_col]
            table[i, :] -= multiplier * table[pivot_row, :]
    return table

def check_stop_condition(table):
    if np.all(table[0, 1:-1] >= 0):
        return True  # 最適解に到達した
    if np.all(table[1:, -1] <= 0):
        return True  # 無制限の問題
    return False

def simplex_method(c, A, b):
    table = initialize_simplex_table(c, A, b)
    while not check_stop_condition(table):
        pivot_row, pivot_col = pivot_selection(table)
        table = pivot_operation(table, pivot_row, pivot_col)
        print("シンプレックス表:")
        print(table)
    return table[0, 0], table[1:, 0]

# 使用例:
c = np.array([2, 1, 0, 0, 0])
A = np.array([[1, 2, 1, 0, 0], [1, 1, 0, 1, 0], [3, 1, 0, 0, 1]])
b = np.array([14, 8, 18])

optimal_value, optimal_solution = simplex_method(c, A, b)
print("最適値:", optimal_value)
print("最適解:", optimal_solution)
