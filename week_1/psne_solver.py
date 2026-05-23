import numpy as np

def  solve_psne(p1_matrix, p2_matrix):

  #Payoff Matrices
  p1 = np.array(p1_matrix, dtype=float)
  p2 = np.array(p2_matrix, dtype=float)

  #Payoff Matrices should be of equal size
  if p1.shape != p2.shape:
    raise ValueError(f"Payoff matrices must have identical dimensions,"
                     f"got {p1.shape} and {p2.shape}.")


  n_rows, n_cols = p1.shape

  p1_col_max = p1.max(axis=0)
  p1_best_response = (p1 == p1_col_max)

  p2_row_max = p2.max(axis=1)
  p2_best_response = (p2 == p2_row_max[:, None])

  psne = p1_best_response & p2_best_response

  rows, cols = np.where(psne)
  psne_list = []

  for i in range(len(rows)):
      psne_list.append((int(rows[i]), int(cols[i])))

  return psne_list

#Example 1

p1 = [
    [3, 1],
    [0, 2]
]

p2 = [
    [2, 1],
    [0, 3]
]

print(solve_psne(p1, p2))

#Example 2

p1 = [
    [4, 2],
    [1, 3]
]

p2 = [
    [4, 1],
    [2, 3]
]

print(solve_psne(p1, p2))

#Example 3

p1 = [
    [5, 1],
    [2, 0]
]

p2 = [
    [4, 3],
    [1, 2]
]

#Example 4

print(solve_psne(p1, p2))

p1 = [
    [1, 2],
    [2, 1]
]

p2 = [
    [2, 1],
    [1, 2]
]

print(solve_psne(p1, p2))
