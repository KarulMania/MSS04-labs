# -*- coding: utf-8 -*-

import cmath

def q_solve(a, b, c):
    if a == 0:
        if b == 0:
            if c == 0:
                return "infinitely many solutions"
            else:
                return "no solution"
        else:
            return [-c / b]

    discriminant = b**2 - 4*a*c
    sqrt_d = cmath.sqrt(discriminant)

    x1 = (-b + sqrt_d) / (2*a)
    x2 = (-b - sqrt_d) / (2*a)

    return [x1] if x1 == x2 else [x1, x2]


print(q_solve(1, -5, 6))   # 2, 3
print(q_solve(1, 2, 1))    # -1
print(q_solve(1, 0, 1))    # complex
print(q_solve(0, 2, -4))   # 2
