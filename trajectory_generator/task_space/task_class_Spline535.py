import numpy as np
import time
import matplotlib.pyplot as plt

class Spline535():
    def __init__(self, time_sequence, control_point):
        self.time_sequence = time_sequence
        self.time_interval = list(time_sequence)
        self.num_rows = len(control_point[:, 0])
        for i in range(len(time_sequence) - 1):
            t1 = time_sequence[i]
            t2 = time_sequence[i + 1]
            self.time_interval.append(t1 + (t2 - t1) / 3)
            self.time_interval.append(t1 + 2 * (t2 - t1) / 3)
        self.time_interval.sort()
        # right side value of the equation
        # self.r_value_x, self.r_value_x, self.r_value_x = [], [], []

    def square_list(self, i, lst):
        # Translate each row into columns with the pattern 6-4-6-6-4-6.
        # Additionally, for the rows with 4 columns, append two zeros at the end to make each row have 6 columns
        # Define the pattern for columns in each row
        ''' i=3 means 3 way points, which is n=5 '''
        if i == 3:
            pattern = [6, 4, 6, 6, 4, 6, 6, 4, 6, 6, 4, 6]
        elif i == 2:
            pattern = [6, 4, 6, 6, 4, 6, 6, 4, 6]
        # Initialize the result list
        result = []

        # Initialize the starting index
        index = 0

        for count in pattern:
            # Extract the current segment from the list
            row = lst[index:index + count]

            # If the current segment has 4 columns, append two zeros to make it 6 columns
            if count == 4:
                row.extend([0, 0])

            # Add the processed row to the result
            result.append(row)

            # Update the index for the next segment
            index += count

        return np.array(result)

    def calc_535(self, a, t):
        # a is coefficients[a0,...a5], for cubic spline, a4=a5=0
        # t is t_current
        pos = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
        vel = a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4
        acc = 2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3
        jerk = 6*a[3] + 24*a[4]*t + 60*a[5]*t**2

        return pos, vel, acc, jerk

    def co_calculate_wp1(self, t11, t12, t_via, t21, t22, tf, given_value):
        # t0 = 0, end time is tf
        # t11, t12 is the start and end time of 3-spline
        # starting point: a 4 equations
        # a0    a1  a2  a3  a4  a5  b0  b1  b2  b3  c0  c1  c2  c3  c4 c5 d0 d1 d2 d3 d4 d5 e0 e1 e2 e3 f0 f1 f2 f3 f4 f5
        M = np.asarray(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             # tf: f  4 equations
             # a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1 c2 c3 c4 c5 d0 d1 d2 d3 d4 d5 e0 e1 e2 e3 f0  f1  f2      f3      f4        f5
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 24 * tf,
              60 * tf ** 2],
             # t11: a and b  4 equations
             # a0  a1    a2      a3          a4          a5           b0       b1       b2            b3    c0 c1 c2 c3 c4 c5 d0 d1 d2 d3 d4 d5 e0 e1 e2 e3 f0 f1 f2 f3 f4 f5
             [1, t11, t11 ** 2, t11 ** 3, t11 ** 4, t11 ** 5, -1, -t11, -t11 ** 2, -t11 ** 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 2 * t11, 3 * t11 ** 2, 4 * t11 ** 3, 5 * t11 ** 4, 0, -1, -2 * t11, -3 * t11 ** 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 2, 6 * t11, 12 * t11 ** 2, 20 * t11 ** 3, 0, 0, -2, -6 * t11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 6, 24 * t11, 60 * t11 ** 2, 0, 0, 0, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             # t12: b and c  4 equations
             # a0 a1 a2 a3 a4 a5  b0   b1      b2        b3          c0  c1      c2      c3          c4          c5          d0 d1 d2 d3 d4 d5 e0 e1 e2 e3 f0 f1 f2 f3 f4 f5
             [0, 0, 0, 0, 0, 0, -1, -t12, -t12 ** 2, -t12 ** 3, 1, t12, t12 ** 2, t12 ** 3, t12 ** 4, t12 ** 5, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, -1, -2 * t12, -3 * t12 ** 2, 0, 1, 2 * t12, 3 * t12 ** 2, 4 * t12 ** 3, 5 * t12 ** 4,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, -2, -6 * t12, 0, 0, 2, 6 * t12, 12 * t12 ** 2, 20 * t12 ** 3, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 6, 24 * t12, 60 * t12 ** 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 0, 0],
             # tvia: c and d  8 equations
             #  a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1      c2          c3          c4              c5              d0      d1      d2      d3              d4              d5              e0 e1 e2 e3 f0 f1 f2 f3 f4 f5
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, t_via, t_via ** 2, t_via ** 3, t_via ** 4, t_via ** 5, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2 * t_via, 3 * t_via ** 2, 4 * t_via ** 3, 5 * t_via ** 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, t_via, t_via ** 2, t_via ** 3, t_via ** 4, t_via ** 5, -1, -t_via, -t_via ** 2, -t_via ** 3, -t_via ** 4, -t_via ** 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2 * t_via, 3 * t_via ** 2, 4 * t_via ** 3, 5 * t_via ** 4, 0, -1, -2 * t_via, -3 * t_via ** 2, -4 * t_via ** 3, -5 * t_via ** 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6 * t_via, 12 * t_via ** 2, 20 * t_via ** 3, 0, 0, -2, -6 * t_via, -12 * t_via ** 2, -20 * t_via ** 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 24 * t_via, 60 * t_via ** 2, 0, 0, 0, -6, -24 * t_via, -60 * t_via ** 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 120 * t_via, 0, 0, 0, 0, -24, -120 * t_via, 0, 0, 0, 0, 0,           0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 0, 0, 0, 0, 0, -120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             # t21 d and e
             # a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1 c2 c3 c4 c5  d0   d1      d2          d3          d4          d5          e0      e1      e2          e3          f0 f1 f2 f3 f4 f5
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, t21, t21 ** 2, t21 ** 3, t21 ** 4, t21 ** 5, -1, -t21, -t21 ** 2, -t21 ** 3, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2 * t21, 3 * t21 ** 2, 4 * t21 ** 3, 5 * t21 ** 4, 0, -1, -2 * t21, -3 * t21 ** 2, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 6 * t21, 12 * t21 ** 2, 20 * t21 ** 3, 0, 0, -2, -6 * t21, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 24 * t21, 60 * t21 ** 2, 0, 0, 0, -6, 0, 0, 0,    0, 0, 0],
             # t22 e and f
             # a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1 c2 c3 c4 c5 d0 d1 d2 d3 d4 d5     e0      e1      e2          e3            f0      f1          f2          f3          f4          f5
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -t22, -t22 ** 2, -t22 ** 3, 1, t22,   t22 ** 2, t22 ** 3, t22 ** 4, t22 ** 5],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2 * t22, -3 * t22 ** 2, 0, 1,     2 * t22, 3 * t22 ** 2, 4 * t22 ** 3, 5 * t22 ** 4],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -6 * t22, 0, 0, 2, 6 * t22,     12 * t22 ** 2, 20 * t22 ** 3],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 0, 0, 6, 24 * t22,        60 * t22 ** 2]
             ])
        return np.linalg.solve(M, given_value)

    def co_calculate_wp2(self, r_value):
        # in total 48 equations
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = self.time_interval
        # n=4, 16*(n-1)=48 equations
        M = [  # t0 = 0 ,4 equations
             [1, 0, 0, 0, 0, 0] + [0]*42,
             [0, 1, 0, 0, 0, 0] + [0]*42,
             [0, 0, 2, 0, 0, 0] + [0]*42,
             [0, 0, 0, 6, 0, 0] + [0]*42,
             # t1: a and b  4 equations
             # a0  a1    a2     a3      a4         a5       b0  b1   b2        b3    c0 c1 c2 c3 c4 c5 d0 d1 d2 d3 d4 d5 e0 e1 e2 e3 f0 f1 f2 f3 f4 f5 g0 g1 g2 g3 g4 g5 h0 h1 h2 h3 i0 i1 i2 i3 i4 i5 j0 j1 j2 j3 j4 j5 k0 k1 k2 k3 l0 l1 l2 l3 l4 l5
             [1, t1,   t1**2, t1**3,    t1**4,    t1**5,   -1, -t1, -t1**2,  -t1**3] + [0]*38,
             [0, 1,    2*t1,  3*t1**2,  4*t1**3,  5*t1**4,  0, -1, -2*t1,    -3*t1**2] + [0]*38,
             [0, 0,    2,     6*t1,    12*t1**2,  20*t1**3, 0,  0, -2,       -6*t1] + [0]*38,
             [0, 0,    0,       6,     24*t1,     60*t1**2, 0,  0,  0,       -6] + [0]*38,
             # t2: b and c, 4 equations
             # a0 a1 a2 a3 a4 a5  b0   b1      b2        b3          c0  c1      c2      c3          c4          c5          d0 d1 d2 d3 d4 d5 e0 e1 e2 e3 f0 f1 f2 f3 f4 f5
             [0]*6 + [-1, -t2, -t2**2, -t2**3, 1, t2, t2**2, t2**3, t2**4, t2**5] + [0]*32,
             [0]*6 + [0, -1, -2*t2, -3*t2**2, 0, 1, 2*t2, 3*t2**2, 4*t2**3, 5*t2**4] + [0]*32,
             [0]*6 + [0, 0, -2, -6*t2, 0, 0, 2, 6*t2, 12*t2**2, 20*t2**3] + [0]*32,
             [0]*6 + [0, 0, 0, -6, 0, 0, 0, 6, 24*t2, 60*t2**2] + [0]*32,
             # t3: c and d, 8 equations
             #  a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1      c2          c3          c4              c5              d0      d1      d2      d3              d4              d5              e0 e1 e2 e3 f0 f1 f2 f3 f4 f5
             [0]*10 + [1, t3,   t3**2, t3**3,   t3**4,      t3**5,    0, 0, 0, 0, 0, 0] + [0]*26,
             [0]*10 + [0, 1,    2*t3,  3*t3**2, 4*t3**3,    5*t3**4,  0, 0, 0, 0, 0, 0] + [0]*26,
             [0]*10 + [1, t3,   t3**2, t3**3,   t3**4,      t3**5,   -1, -t3, -t3**2, -t3**3, -t3**4, -t3**5] + [0]*26,
             [0]*10 + [0, 1,    2*t3,  3*t3**2, 4*t3**3,    5*t3**4,  0, -1, -2*t3, -3*t3**2, -4*t3**3, -5*t3**4] + [0]*26,
             [0]*10 + [0, 0,    2,     6*t3,    12*t3**2,   20*t3**3, 0, 0, -2, -6*t3, -12*t3**2, -20*t3**3] + [0]*26,
             [0]*10 + [0, 0, 0, 6,     24*t3,   60*t3**2,   0, 0, 0, -6, -24*t3, -60*t3**2] + [0]*26,
             [0]*10 + [0, 0, 0, 0,     24,      120*t3,     0, 0, 0, 0, -24, -120*t3] + [0]*26,
             [0]*10 + [0, 0, 0, 0,     0,       120,        0, 0, 0, 0, 0,   -120] + [0]*26,
             # t4 d and e, 4 equations
             # a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1 c2 c3 c4 c5  d0   d1      d2          d3          d4          d5          e0      e1      e2          e3          f0 f1 f2 f3 f4 f5
             [0]*16 + [1, t4, t4**2, t4**3,     t4**4,      t4**5,   -1, -t4, -t4**2, -t4**3] + [0]*22,
             [0]*16 + [0, 1, 2*t4,   3*t4**2,   4*t4**3,    5*t4**4,  0, -1, -2*t4, -3*t4**2] + [0]*22,
             [0]*16 + [0, 0, 2,      6*t4,      12*t4**2,   20*t4**3, 0,  0, -2, -6*t4] + [0]*22,
             [0]*16 + [0, 0, 0,      6,         24*t4,      60*t4**2, 0,  0,  0, -6] + [0]*22,
             # t5 e and f, 4 equations
             # a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1 c2 c3 c4 c5 d0 d1 d2 d3 d4 d5     e0      e1      e2          e3            f0      f1          f2          f3          f4          f5
             [0]*22 + [-1, -t5, -t5**2, -t5**3,     1, t5,   t5**2, t5**3,   t5**4, t5**5] + [0]*16,
             [0]*22 + [0, -1,   -2*t5, -3*t5**2,    0, 1,    2*t5,  3*t5**2, 4*t5**3, 5*t5**4] + [0]*16,
             [0]*22 + [0, 0,    -2,     -6*t5,      0, 0,    2,     6*t5,    12*t5**2, 20*t5**3] + [0]*16,
             [0]*22 + [0, 0,    0,      -6,         0, 0,    0,     6,       24*t5,  60*t5**2] + [0]*16,
             # t6: f g  8 equations
             # a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1 c2 c3 c4 c5 d0 d1 d2 d3 d4 d5 e0 e1 e2 e3 f0  f1  f2      f3      f4        f5                       g0 g1 g2 g3 g4 g5 h0 h1 h2 h3 i0 i1 i2 i3 i4 i5 j0 j1 j2 j3 j4 j5 k0 k1 k2 k3 l0 l1 l2 l3 l4 l5
             [0]*26 + [1, t6,     t6**2, t6**3,     t6**4,       t6**5,       0,      0,      0,      0,              0,      0] +     [0]*10,
             [0]*26 + [0, 1,      2*t6,  3*t6**2,   4*t6**3,     5*t6**4,     0,      0,      0,      0,              0,       0] +    [0]*10,
             [0]*26 + [1, t6,     t6**2, t6**3,     t6**4,       t6**5,       -1, -t6, -t6**2,  -t6**3,      -t6**4,      -t6**5] +    [0]*10,
             [0]*26 + [0, 1,      2*t6,  3*t6**2,   4*t6**3,     5*t6**4,     0,  -1,     -2*t6,   -3*t6**2,  -4*t6**3,    -5*t6**4] + [0]*10,
             [0]*26 + [0, 0,      2,     6*t6,      12*t6**2,    20*t6**3,    0,  0,      -2,     -6*t6,     -12*t6**2,   -20*t6**3] + [0]*10,
             [0]*26 + [0, 0,      0,        6,            24*t6,       60*t6**2,    0,  0,      0,     -6,   -24*t6,      -60*t6**2] + [0]*10,
             [0]*26 + [0, 0,      0,        0,            24,             120*t6,    0,  0,    0,     0,    -24,    -120*t6] +         [0]*10,
             [0]*26 + [0, 0,      0,        0,            0,              120,      0,  0,      0,   0,      0,     -120] +            [0]*10,
             # t7: gh 4 equations
             [0]*32 + [1, t7,  t7 ** 2,    t7 ** 3, t7 ** 4,   t7**5,    -1,     -t7,    -t7**2,     -t7**3] +   [0]*6,
             [0]*32 + [0, 1,   2*t7,       3*t7**2, 4*t7**3,   5*t7**4,  0,      -1,     -2*t7,      -3*t7**2] + [0]*6,
             [0]*32 + [0, 0,   2,          6*t7,    12*t7**2,  20*t7**3, 0,      0,       -2,        -6*t7]    + [0]*6,
             [0]*32 + [0, 0,   0,          6,       24*t7,     60*t7**2, 0,      0,       0,          -6]      + [0]*6,
             # t8: hi 4 equations
             [0]*38 + [-1, -t8,  -t8**2,   -t8**3,     1,  t8,    t8**2, t8**3,     t8**4,     t8**5] + [0]*0,
             [0]*38 + [0,  -1,    -2*t8,    -3*t8**2,   0,  1,      2*t8,  3*t8**2,   4*t8**3,   5*t8**4] + [0]*0,
             [0]*38 + [0,  0,     -2,        -6*t8,      0,  0,      2,      6*t8,      12*t8**2,  20*t8**3] + [0]*0,
             [0]*38 + [0,  0,     0,         -6,          0,  0,      0,      6,          24*t8,     60*t8**2] + [0]*0,
             # t9: ij 8 equations
            [0] * 42 + [1, t9, t9**2, t9**3, t9**4, t9**5],
            [0] * 42 + [0, 1, 2*t9, 3*t9**2, 4*t9**3, 5*t9**4],
            [0] * 42 + [0, 0, 2, 6*t9, 12*t9**2, 20*t9**3],
            [0] * 42 + [0, 0, 0, 6, 24*t9, 60*t9**2]
        ]
        return self.square_list(2, list(np.linalg.solve(M, r_value)))
    def co_calculate_wp3(self, r_value):
        # r_value: right side value of the equation
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12 = self.time_interval
        M = [  # t0 = 0 ,4 equations
             [1, 0, 0, 0, 0, 0] + [0]*58,
             [0, 1, 0, 0, 0, 0] + [0]*58,
             [0, 0, 2, 0, 0, 0] + [0]*58,
             [0, 0, 0, 6, 0, 0] + [0]*58,
             # t1: a and b  4 equations
             # a0  a1    a2     a3      a4         a5       b0  b1   b2        b3    c0 c1 c2 c3 c4 c5 d0 d1 d2 d3 d4 d5 e0 e1 e2 e3 f0 f1 f2 f3 f4 f5 g0 g1 g2 g3 g4 g5 h0 h1 h2 h3 i0 i1 i2 i3 i4 i5 j0 j1 j2 j3 j4 j5 k0 k1 k2 k3 l0 l1 l2 l3 l4 l5
             [1, t1,   t1**2, t1**3,    t1**4,    t1**5,   -1, -t1, -t1**2,  -t1**3] + [0]*54,
             [0, 1,    2*t1,  3*t1**2,  4*t1**3,  5*t1**4,  0, -1, -2*t1,    -3*t1**2] + [0]*54,
             [0, 0,    2,     6*t1,    12*t1**2,  20*t1**3, 0,  0, -2,       -6*t1] + [0]*54,
             [0, 0,    0,       6,     24*t1,     60*t1**2, 0,  0,  0,       -6] + [0]*54,
             # t2: b and c, 4 equations
             # a0 a1 a2 a3 a4 a5  b0   b1      b2        b3          c0  c1      c2      c3          c4          c5          d0 d1 d2 d3 d4 d5 e0 e1 e2 e3 f0 f1 f2 f3 f4 f5
             [0]*6 + [-1, -t2, -t2**2, -t2**3, 1, t2, t2**2, t2**3, t2**4, t2**5] + [0]*48,
             [0]*6 + [0, -1, -2*t2, -3*t2**2, 0, 1, 2*t2, 3*t2**2, 4*t2**3, 5*t2**4] + [0]*48,
             [0]*6 + [0, 0, -2, -6*t2, 0, 0, 2, 6*t2, 12*t2**2, 20*t2**3] + [0]*48,
             [0]*6 + [0, 0, 0, -6, 0, 0, 0, 6, 24*t2, 60*t2**2] + [0]*48,
             # t3: c and d, 8 equations
             #  a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1      c2          c3          c4              c5              d0      d1      d2      d3              d4              d5              e0 e1 e2 e3 f0 f1 f2 f3 f4 f5
             [0]*10 + [1, t3,   t3**2, t3**3,   t3**4,      t3**5,    0, 0, 0, 0, 0, 0] + [0]*42,
             [0]*10 + [0, 1,    2*t3,  3*t3**2, 4*t3**3,    5*t3**4,  0, 0, 0, 0, 0, 0] + [0]*42,
             [0]*10 + [1, t3,   t3**2, t3**3,   t3**4,      t3**5,   -1, -t3, -t3**2, -t3**3, -t3**4, -t3**5] + [0]*42,
             [0]*10 + [0, 1,    2*t3,  3*t3**2, 4*t3**3,    5*t3**4,  0, -1, -2*t3, -3*t3**2, -4*t3**3, -5*t3**4] + [0]*42,
             [0]*10 + [0, 0,    2,     6*t3,    12*t3**2,   20*t3**3, 0, 0, -2, -6*t3, -12*t3**2, -20*t3**3] + [0]*42,
             [0]*10 + [0, 0, 0, 6,     24*t3,   60*t3**2,   0, 0, 0, -6, -24*t3, -60*t3**2] + [0]*42,
             [0]*10 + [0, 0, 0, 0,     24,      120*t3,     0, 0, 0, 0, -24, -120*t3] + [0]*42,
             [0]*10 + [0, 0, 0, 0,     0,       120,        0, 0, 0, 0, 0,   -120] + [0]*42,
             # t4 d and e, 4 equations
             # a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1 c2 c3 c4 c5  d0   d1      d2          d3          d4          d5          e0      e1      e2          e3          f0 f1 f2 f3 f4 f5
             [0]*16 + [1, t4, t4**2, t4**3,     t4**4,      t4**5,   -1, -t4, -t4**2, -t4**3] + [0]*38,
             [0]*16 + [0, 1, 2*t4,   3*t4**2,   4*t4**3,    5*t4**4,  0, -1, -2*t4, -3*t4**2] + [0]*38,
             [0]*16 + [0, 0, 2,      6*t4,      12*t4**2,   20*t4**3, 0,  0, -2, -6*t4] + [0]*38,
             [0]*16 + [0, 0, 0,      6,         24*t4,      60*t4**2, 0,  0,  0, -6] + [0]*38,
             # t5 e and f, 4 equations
             # a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1 c2 c3 c4 c5 d0 d1 d2 d3 d4 d5     e0      e1      e2          e3            f0      f1          f2          f3          f4          f5
             [0]*22 + [-1, -t5, -t5**2, -t5**3,     1, t5,   t5**2, t5**3,   t5**4, t5**5] + [0]*32,
             [0]*22 + [0, -1,   -2*t5, -3*t5**2,    0, 1,    2*t5,  3*t5**2, 4*t5**3, 5*t5**4] + [0]*32,
             [0]*22 + [0, 0,    -2,     -6*t5,      0, 0,    2,     6*t5,    12*t5**2, 20*t5**3] + [0]*32,
             [0]*22 + [0, 0,    0,      -6,         0, 0,    0,     6,       24*t5,  60*t5**2] + [0]*32,
             # t6: f g  8 equations
             # a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 c0 c1 c2 c3 c4 c5 d0 d1 d2 d3 d4 d5 e0 e1 e2 e3 f0  f1  f2      f3      f4        f5                       g0 g1 g2 g3 g4 g5 h0 h1 h2 h3 i0 i1 i2 i3 i4 i5 j0 j1 j2 j3 j4 j5 k0 k1 k2 k3 l0 l1 l2 l3 l4 l5
             [0]*26 + [1, t6,     t6**2, t6**3,     t6**4,       t6**5,       0,      0,      0,      0,              0,      0] +     [0]*26,
             [0]*26 + [0, 1,      2*t6,  3*t6**2,   4*t6**3,     5*t6**4,     0,      0,      0,      0,              0,       0] +    [0]*26,
             [0]*26 + [1, t6,     t6**2, t6**3,     t6**4,       t6**5,       -1, -t6, -t6**2,  -t6**3,      -t6**4,      -t6**5] +    [0]*26,
             [0]*26 + [0, 1,      2*t6,  3*t6**2,   4*t6**3,     5*t6**4,     0,  -1,     -2*t6,   -3*t6**2,  -4*t6**3,    -5*t6**4] + [0]*26,
             [0]*26 + [0, 0,      2,     6*t6,      12*t6**2,    20*t6**3,    0,  0,      -2,     -6*t6,     -12*t6**2,   -20*t6**3] + [0]*26,
             [0]*26 + [0, 0,      0,        6,            24*t6,       60*t6**2,    0,  0,      0,     -6,   -24*t6,      -60*t6**2] + [0]*26,
             [0]*26 + [0, 0,      0,        0,            24,             120*t6,    0,  0,    0,     0,    -24,    -120*t6] +         [0]*26,
             [0]*26 + [0, 0,      0,        0,            0,              120,      0,  0,      0,   0,      0,     -120] +            [0]*26,

             # t7: gh 4 equations
             [0]*32 + [1, t7,  t7 ** 2,    t7 ** 3, t7 ** 4,   t7**5,    -1,     -t7,    -t7**2,     -t7**3] +   [0]*22,
             [0]*32 + [0, 1,   2*t7,       3*t7**2, 4*t7**3,   5*t7**4,  0,      -1,     -2*t7,      -3*t7**2] + [0]*22,
             [0]*32 + [0, 0,   2,          6*t7,    12*t7**2,  20*t7**3, 0,      0,       -2,        -6*t7]    + [0]*22,
             [0]*32 + [0, 0,   0,          6,       24*t7,     60*t7**2, 0,      0,       0,          -6]      + [0]*22,
             # t8: hi 4 equations
             [0]*38 + [-1, -t8,  -t8**2,   -t8**3,     1,  t8,    t8**2, t8**3,     t8**4,     t8**5] + [0]*16,
             [0]*38 + [0,  -1,    -2*t8,    -3*t8**2,   0,  1,      2*t8,  3*t8**2,   4*t8**3,   5*t8**4] + [0]*16,
             [0]*38 + [0,  0,     -2,        -6*t8,      0,  0,      2,      6*t8,      12*t8**2,  20*t8**3] + [0]*16,
             [0]*38 + [0,  0,     0,         -6,          0,  0,      0,      6,          24*t8,     60*t8**2] + [0]*16,
             # t9: ij 8 equations
             [0]*42 + [1, t9,     t9**2, t9**3,     t9**4,       t9**5,       0,      0,      0,      0,              0,      0] +     [0]*10,
             [0]*42 + [0, 1,      2*t9,  3*t9**2,   4*t9**3,     5*t9**4,     0,      0,      0,      0,              0,       0] +    [0]*10,
             [0]*42 + [1, t9,     t9**2, t9**3,     t9**4,       t9**5,       -1, -t9, -t9**2,  -t9**3,      -t9**4,      -t9**5] +    [0]*10,
             [0]*42 + [0, 1,      2*t9,  3*t9**2,   4*t9**3,     5*t9**4,     0,  -1,     -2*t9,   -3*t9**2,  -4*t9**3,    -5*t9**4] + [0]*10,
             [0]*42 + [0, 0,      2,     6*t9,      12*t9**2,    20*t9**3,    0,  0,      -2,     -6*t9,     -12*t9**2,   -20*t9**3] + [0]*10,
             [0]*42 + [0, 0,      0,        6,            24*t9,       60*t9**2,    0,  0,      0,     -6,   -24*t9,      -60*t9**2] + [0]*10,
             [0]*42 + [0, 0,      0,        0,            24,             120*t9,    0,  0,    0,     0,    -24,    -120*t9] +         [0]*10,
             [0]*42 + [0, 0,      0,        0,            0,              120,      0,  0,      0,   0,      0,     -120] +            [0]*10,
             # t10: jk 4 equations
             [0]*48 + [1, t10,  t10 ** 2,    t10 ** 3, t10 ** 4,   t10**5,    -1,     -t10,    -t10**2,     -t10**3] +   [0]*6,
             [0]*48 + [0, 1,   2*t10,       3*t10**2, 4*t10**3,   5*t10**4,  0,      -1,     -2*t10,      -3*t10**2] + [0]*6,
             [0]*48 + [0, 0,   2,          6*t10,    12*t10**2,  20*t10**3, 0,      0,       -2,        -6*t10]    + [0]*6,
             [0]*48 + [0, 0,   0,          6,       24*t10,     60*t10**2, 0,      0,       0,          -6]      + [0]*6,
             # t11: kl 4 equations
             [0]*54 + [-1, -t11,  -t11**2,   -t11**3,     1,  t11,    t11**2, t11**3,     t11**4,     t11**5],
             [0]*54 + [0,  -1,    -2*t11,    -3*t11**2,   0,  1,      2*t11,  3*t11**2,   4*t11**3,   5*t11**4],
             [0]*54 + [0,  0,     -2,        -6*t11,      0,  0,      2,      6*t11,      12*t11**2,  20*t11**3],
             [0]*54 + [0,  0,     0,         -6,          0,  0,      0,      6,          24*t11,     60*t11**2],
             # tf = t12: l 4 equations
             [0]*58 + [1, t12, t12**2,  t12**3,  t12**4,    t12**5],
             [0]*58 + [0, 1,  2*t12,   3*t12**2, 4*t12**3,  5*t12**4],
             [0]*58 + [0, 0,  2,      6*t12,    12*t12**2, 20*t12**3],
             [0]*58 + [0, 0,  0,      6,       24*t12,    60*t12**2]
             ]

        return self.square_list(3, list(np.linalg.solve(M, r_value)))




if __name__ == '__main__':
    # -----------------------initialization---------------------------
    pos_x, pos_y, pos_z, v_x, v_y, v_z, a_x, a_y, a_z = [], [], [], [], [], [], [], [], []
    planned_px, planned_py, planned_pz = [], [], []
    planned_ax, planned_ay, planned_az, planned_vx, planned_vy, planned_vz = [], [], [], [], [], []
    planned_jx = []
    time_range = []
    # # -----------------------data set 1---------------------------
    # input_x = np.array([4.15, 42.98, -82.06, 301.28, 227.88]).reshape(5, 1)
    # input_y = np.array([-261.03, -328.32, -276.68, -131.99, -300.54]).reshape(5, 1)
    # input_z = np.array([0.441, 192.91, 313.66, 206.41, 358.10]).reshape(5, 1)
    # # -----------------------data set 2---------------------------
    # input_x = np.array([4.15, 42.98, 301.28, 227.88]).reshape(4, 1)
    # input_y = np.array([-261.03, -328.32, -276.68, -300.54]).reshape(4, 1)
    # input_z = np.array([192.91, 313.66, 206.41, 358.10]).reshape(4, 1)
    # control_point = np.hstack((input_x, input_y, input_z))
    # time_sequence = np.linspace(0, 5, 4)
    # t_final = time_sequence[-1]
    ''' define numer of control points'''
    num_rows = 4

    if num_rows == 5:
        # -----------------------data set 1---------------------------
        input_x = np.array([4.15, 42.98, -82.06, 301.28, 227.88]).reshape(5, 1)
        input_y = np.array([-261.03, -328.32, -276.68, -131.99, -300.54]).reshape(5, 1)
        input_z = np.array([0.441, 192.91, 313.66, 206.41, 358.10]).reshape(5, 1)
        control_point = np.hstack((input_x, input_y, input_z))
        time_sequence = np.linspace(0, 5, 4)
        t_final = time_sequence[-1]
        num_rows = np.shape(input_x)[0]
        # -----------------------user defined value---------------------------
        # start pos, vel, acc, jerk
        qx_0, vx_0, ax_0, jx_0 = list(input_x[0]), 0, 0, 0
        # way point pos, vel
        qx_1, vx_1, qx_2, vx_2, qx_3, vx_3 = list(input_x[1]), 0, list(input_x[2]), 0, list(input_x[3]), 0
        # end pos, vel, acc, jerk
        qx_4, vx_4, ax_4, jx_4 = list(input_x[4]), 0, 0, 0
        # right side value of coefficient equations:  a0 + a1t + ...+ a5t**5 = r_value
        r_value_x = [qx_0, vx_0, ax_0, jx_0] + [0] * 8 + [qx_1, vx_1] + [0] * 6 + \
                    [0] * 8 + [qx_2, vx_2] + [0] * 6 + \
                    [0] * 8 + [qx_3, vx_3] + [0] * 6 + \
                    [0] * 8 + [qx_4, vx_4, ax_4, jx_4]
        r_value_x = [item[0] if isinstance(item, list) else item for item in r_value_x]
        # -----------------------read coefficients from class---------------------------------------
        planner = Spline535(time_sequence, control_point)
        ''' for 2 way points(n=4), choose co_calculate_wp2 '''
        M_x = planner.co_calculate_wp3(r_value_x)
        # M_x = planner.square_list(list(M_x))  # (M_x.shape) = (12, 6)
        # time_interval= [0.0, 0.4166666666666667, 0.8333333333333334, 1.25, 1.6666666666666667, 2.0833333333333335,
        # 2.5, 2.9166666666666665, 3.3333333333333335, 3.75, 4.166666666666667, 4.583333333333333, 5.0]

        t_start = time.time()
        t_current = time.time() - t_start
        while t_current < t_final:
            t_current = time.time() - t_start
            for i in range(3 * planner.num_rows - 3):  # n=5-->len(time_interval)=13(3n-2), therefore max=12=3n-3
                if planner.time_interval[i] <= t_current < planner.time_interval[i + 1]:
                    t_current = time.time() - t_start
                    px, vx, ax, jx = planner.calc_535(M_x[i], t_current)
                    break
            planned_px.append(px), planned_vx.append(vx), planned_ax.append(ax)
            time_range.append(t_current)

        plotter = 1
        if plotter:
            # ------------pos-------------
            plt.figure(1)
            plt.title('position x')
            plt.plot(time_sequence, input_x, '*', color='green', label='control point x')
            plt.plot(time_range, planned_px, label='X position')
            plt.plot(time_range, planned_vx, label='x velocity [m/s]')
            # plt.plot(time_range, planned_ax, label='x acceleration [m/s]')
            plt.legend(), plt.grid()
            plt.ylabel('Position [m]')
            plt.xlabel('Time [s]')

            # plt.figure(2)
            # plt.title('position y')
            # plt.plot(time_sequence, input_y, '*', color='green', label='control point x')
            # plt.plot(time_range, planned_py, label='y position [mm]')
            # plt.legend(), plt.grid()
            # plt.ylabel('Position [mm]')
            # plt.xlabel('Time [s]')
            #
            # plt.figure(3)
            # plt.title('position z')
            # plt.plot(time_sequence, input_z, '*', color='green', label='control point z')
            # plt.plot(time_range, planned_pz, label='z position [mm]')
            # plt.legend(), plt.grid()
            # plt.ylabel('Position [mm]')
            # plt.xlabel('Time [s]')
            # ----------vel----------
            # ----------acc----------
            plt.show()

    elif num_rows == 4:
        # -----------------------data set 2---------------------------
        input_x = np.array([5.05, 42.98, 150.15, 277.88]).reshape(4, 1) * 0.001
        input_y = np.array([-260.95, -250.25, -276.68, -300.54]).reshape(4, 1) * 0.001
        input_z = np.array([440.64, 380.38, 317.05, 358.10]).reshape(4, 1) * 0.001
        control_point = np.hstack((input_x, input_y, input_z))
        time_sequence = np.linspace(0, 5, 4)
        t_final = time_sequence[-1]
        num_rows = np.shape(input_x)[0]


        planner = Spline535(time_sequence, control_point)
        # -----------------------user defined value---------------------------
        # ----------------------- x axis ---------------------------
        '''define value at start point: pos(from input_xyz), vel, acc ,jerk'''
        qx_0, vx_0, ax_0, jx_0 = list(input_x[0]), 0, 0, 0
        '''define value at way point: pos(from input_xyz), vel'''
        qx_1, vx_1, qx_2, vx_2 = list(input_x[1]), 0, list(input_x[2]), 0
        # end pos, vel, acc, jerk
        '''define value at end point: pos(from input_xyz), vel, acc ,jerk'''
        qx_3, vx_3, ax_3, jx_3 = list(input_x[-1]), 0, 0, 0
        # right side value of coefficient equations:  a0 + a1t + ...+ a5t**5 = r_value
        r_value_x = [qx_0, vx_0, ax_0, jx_0] + \
                    [0] * 8 + [qx_1, vx_1] + [0] * 6 + \
                    [0] * 8 + [qx_2, vx_2] + [0] * 6 + \
                    [0] * 8 + [qx_3, vx_3, ax_3, jx_3]
        r_value_x = [item[0] if isinstance(item, list) else item for item in r_value_x]
        # ----------------------- y axis ---------------------------
        '''define value at start point: pos(from input_xyz), vel, acc ,jerk'''
        qy_0, vy_0, ay_0, jy_0 = list(input_y[0]), 0, 0, 0
        '''define value at way point: pos(from input_xyz), vel'''
        qy_1, vy_1, qy_2, vy_2 = list(input_y[1]), 0, list(input_y[2]), 0
        # end pos, vel, acc, jerk
        '''define value at end point: pos(from input_xyz), vel, acc ,jerk'''
        qy_3, vy_3, ay_3, jy_3 = list(input_y[-1]), 0, 0, 0
        # right side value of coefficient equations:  a0 + a1t + ...+ a5t**5 = r_value
        r_value_y = [qy_0, vy_0, ay_0, jy_0] + \
                    [0] * 8 + [qy_1, vy_1] + [0] * 6 + \
                    [0] * 8 + [qy_2, vy_2] + [0] * 6 + \
                    [0] * 8 + [qy_3, vy_3, ay_3, jy_3]
        r_value_y = [item[0] if isinstance(item, list) else item for item in r_value_y]
        '''define value at start point: pos(from input_xyz), vel, acc ,jerk'''
        qz_0, vz_0, az_0, jz_0 = list(input_z[0]), 0, 0, 0
        '''define value at waz point: pos(from input_xyz), vel'''
        qz_1, vz_1, qz_2, vz_2 = list(input_z[1]), 0, list(input_z[2]), 0
        # end pos, vel, acc, jerk
        '''define value at end point: pos(from input_xyz), vel, acc ,jerk'''
        qz_3, vz_3, az_3, jz_3 = list(input_z[-1]), 0, 0, 0
        # right side value of coefficient equations:  a0 + a1t + ...+ a5t**5 = r_value
        r_value_z = [qz_0, vz_0, az_0, jz_0] + \
                    [0] * 8 + [qz_1, vz_1] + [0] * 6 + \
                    [0] * 8 + [qz_2, vz_2] + [0] * 6 + \
                    [0] * 8 + [qz_3, vz_3, az_3, jz_3]
        r_value_z = [item[0] if isinstance(item, list) else item for item in r_value_z]

        # -----------------------read coefficients from class---------------------------------------
        '''for 2 way points(n=4), choose co_calculate_wp2'''
        M_x = planner.co_calculate_wp2(r_value_x)
        # M_x = planner.square_list(list(M_x))  # (M_x.shape) = (12, 6)
        # time_interval= [0.0, 0.4166666666666667, 0.8333333333333334, 1.25, 1.6666666666666667, 2.0833333333333335,
        # 2.5, 2.9166666666666665, 3.3333333333333335, 3.75, 4.166666666666667, 4.583333333333333, 5.0]

        t_start = time.time()
        t_current = time.time() - t_start
        while t_current < t_final:
            t_current = time.time() - t_start
            for i in range(3 * planner.num_rows - 3):  # n=5-->len(time_interval)=13(3n-2), therefore max=12=3n-3
                if planner.time_interval[i] <= t_current < planner.time_interval[i + 1]:
                    t_current = time.time() - t_start
                    px, vx, ax, jx = planner.calc_535(M_x[i], t_current)
                    break
            planned_px.append(px), planned_vx.append(vx), planned_ax.append(ax),planned_jx.append(jx)
            time_range.append(t_current)

        plotter = 1
        if plotter:
            # ------------pos-------------
            plt.figure(1)
            plt.title('5-3-5 Spline')
            plt.plot(time_sequence, input_x, '*', color='green', label='control point')
            plt.plot(time_range, planned_px, label='position [m]')
            plt.plot(time_range, planned_vx, label='velocity [m/s]')
            plt.plot(time_range, planned_ax, label='acceleration [m/s^2]')
            plt.plot(time_range, planned_jx, label='jerk [m/s^3]')
            plt.legend(), plt.grid()
            plt.ylabel('Value [m]')
            plt.xlabel('Time [s]')

            # plt.figure(2)
            # plt.title('position y')
            # plt.plot(time_sequence, input_y, '*', color='green', label='control point x')
            # plt.plot(time_range, planned_py, label='y position [mm]')
            # plt.legend(), plt.grid()
            # plt.ylabel('Position [mm]')
            # plt.xlabel('Time [s]')
            #
            # plt.figure(3)
            # plt.title('position z')
            # plt.plot(time_sequence, input_z, '*', color='green', label='control point z')
            # plt.plot(time_range, planned_pz, label='z position [mm]')
            # plt.legend(), plt.grid()
            # plt.ylabel('Position [mm]')
            # plt.xlabel('Time [s]')
            # ----------vel----------
            # ----------acc----------
            plt.show()



