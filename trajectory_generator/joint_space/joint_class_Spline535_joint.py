import numpy as np
import time
import matplotlib.pyplot as plt

class Spline535():
    def __init__(self, time_sequence, control_point, d_q_cp, d2_q_cp, d3_q_cp):
        self.time_sequence = time_sequence
        self.time_interval = list(time_sequence)
        self.num_rows = len(control_point[:, 0])
        for i in range(len(time_sequence) - 1):
            t1 = time_sequence[i]
            t2 = time_sequence[i + 1]
            self.time_interval.append(t1 + (t2 - t1) / 3)
            self.time_interval.append(t1 + 2 * (t2 - t1) / 3)
        self.time_interval.sort()
        self.d_q_cp = d_q_cp
        self.d2_q_cp = d2_q_cp
        self.d3_q_cp = d3_q_cp
        # right side value of the equation
        # self.r_value_x, self.r_value_x, self.r_value_x = [], [], []
    def square_list(self, k, lst):
        '''
        # Translate each row into columns with the pattern 6-4-6-6-4-6.
        # Additionally, for the rows with 4 columns, append two zeros at the end to make each row have 6 columns
        # Define the pattern for columns in each row

        k=3 means 3 way points, which is n=5
        6-4-6 matches the number of coefficients of 5-3-5 polynomial
        for 3-order polynomial, it will be extended to a0, a1, a2, a3, a4, a5 , with a4=a5=0
        '''
        if k == 3:
            pattern = [6, 4, 6, 6, 4, 6, 6, 4, 6, 6, 4, 6]
        elif k == 2:
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

    def co_calculate_wp2(self, cp: [list]):
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
        r_value = self.calc_r_value(cp)
        # r_value = list(r_value)
        # print(r_value)
        return self.square_list(2, list(np.linalg.solve(M, r_value)))

    def calc_r_value(self, cp):
        # -----------------------user defined value---------------------------
        ''' define value at start point: pos(from input_xyz), vel, acc ,jerk '''
        q0, v0, a0, j0 = cp[0], self.d_q_cp[0], self.d2_q_cp[0], self.d3_q_cp[0]

        '''define value at way point: pos(from input_xyz), vel'''
        q1, v1 = cp[1], self.d_q_cp[1]
        q2, v2 = cp[2], self.d_q_cp[2]

        '''define value at end point: pos(from input_xyz), vel, acc ,jerk'''
        q3, v3, a3, j3 = cp[-1], self.d_q_cp[-1], self.d2_q_cp[-1], self.d3_q_cp[-1]

        # right side value of coefficient equations:  a0 + a1t + ...+ a5t**5 = r_value
        r_value = [q0, v0, a0, j0] + \
                  [0] * 8 + [q1, v1] + [0] * 6 + \
                  [0] * 8 + [q2, v2] + [0] * 6 + \
                  [0] * 8 + [q3, v3, a3, j3]

        # turn every element in r_value into type np.ndarray
        r_value = [np.array(item) if not isinstance(item, np.ndarray) else item for item in r_value]
        r_value = np.hstack(r_value)
        return r_value

if __name__ == '__main__':
    # -----------------------initialization---------------------------
    pos_x, pos_y, pos_z, v_x, v_y, v_z, a_x, a_y, a_z = [], [], [], [], [], [], [], [], []
    planned_q0, planned_py, planned_pz = [], [], []
    planned_d2_q0, planned_ay, planned_az, planned_d_q0, planned_vy, planned_vz = [], [], [], [], [], []
    time_range = []
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

        # -----------------------read coefficients from class---------------------------------------
        planner = Spline535(time_sequence, control_point)
        r_value_q0 = planner.calc_r_value(input_x)
        ''' for 2 way points(n=4), choose co_calculate_wp2 '''
        M_q0 = planner.co_calculate_wp3(r_value_q0)
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
                    q0, d_q0, d2_q0, d3_q0 = planner.calc_535(M_q0[i], t_current)
                    break
            planned_q0.append(q0), planned_d_q0.append(d_q0), planned_d2_q0.append(d2_q0)
            time_range.append(t_current)

        plotter = 1
        if plotter:
            # ------------pos-------------
            plt.figure(1)
            plt.title('position x')
            plt.plot(time_sequence, input_x, '*', color='green', label='control point x')
            plt.plot(time_range, planned_q0, label='X position')
            plt.plot(time_range, planned_d_q0, label='x velocity [m/s]')
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
        # -----------------------initialization---------------------------
        planned_q0, planned_q1, planned_q2, planned_q3, planned_q4, planned_q5 = [], [], [], [], [], []
        planned_d_q0, planned_d_q1, planned_d_q2, planned_d_q3, planned_d_q4, planned_d_q5 = [], [], [], [], [], []
        planned_d2_q0, planned_d2_q1, planned_d2_q2, planned_d2_q3, planned_d2_q4, planned_d2_q5 = [], [], [], [], [], []
        time_range = []
        # -----------------------data set ---------------------------
        # control point at TCP postion , unit:m
        input_x = np.array([5.05, 42.98, 301.28, 277.88]).reshape(4, 1) * 0.001
        input_y = np.array([-260.95, -320.32, -276.68, -300.54]).reshape(4, 1) * 0.001
        input_z = np.array([440.64, 313.66, 317.05, 358.10]).reshape(4, 1) * 0.001
        rotation_x = [0.094, 0.094, 0.094, 0.094]
        rotation_y = [-3.035, -3.04, -3.04, -3.04]
        rotation_z = [0.706, 0.706, 0.706, 0.706]
        # this is joint angle in radian
        # 0.01745=2*pi/360 from degree to rad
        input_q0 = np.array([-52.08, -40.55, -29.56, -22.91]).reshape(4, 1) * 0.01745
        input_q1 = np.array([-69.89, -62.23, -78.51, -105.22]).reshape(4, 1) * 0.01745
        input_q2 = np.array([-78.37, -98.58, -104.31, -64.42]).reshape(4, 1) * 0.01745
        input_q3 = np.array([-100.74, -91.70, -73.82, -89.82]).reshape(4, 1) * 0.01745
        input_q4 = np.array([74.04, 70.23, 67.29, 65.92]).reshape(4, 1) * 0.01745
        input_q5 = np.array([31.40, 42.84, 54.21, 61.29]).reshape(4, 1) * 0.01745
        
        
        control_point = np.hstack((input_q0, input_q1, input_q2, input_q3, input_q4, input_q5))
        time_sequence = np.linspace(0, 9, len(input_q0))
        t_final = time_sequence[-1]
        # define vel, acc and jerk at control points
        d_q_cp, d2_q_cp, d3_q_cp = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
        # -----------------------read coefficients from class---------------------------------------
        '''for 2 way points(n=4), choose co_calculate_wp2'''
        planner = Spline535(time_sequence, control_point, d_q_cp, d2_q_cp, d3_q_cp)
        M_q0 = planner.co_calculate_wp2(input_q0)
        M_q1 = planner.co_calculate_wp2(input_q1)
        M_q2 = planner.co_calculate_wp2(input_q2)
        M_q3 = planner.co_calculate_wp2(input_q3)
        M_q4 = planner.co_calculate_wp2(input_q4)
        M_q5 = planner.co_calculate_wp2(input_q5)
        # M_q = planner.square_list(list(M_q))  # (M_q.shape) = (12, 6)
        t_start = time.time()
        t_current = time.time() - t_start
        while t_current < t_final:
            t_current = time.time() - t_start
            # len(planner.time_interval) = 10: element number = 10, therefore 9 interval
            for i in range(3 * planner.num_rows - 3):  # n=4 --> len(time_interval)=10=(3n-2), therefore max=9=3n-3
                if planner.time_interval[i] <= t_current < planner.time_interval[i + 1]:
                    t_current = time.time() - t_start
                    q0, d_q0, d2_q0, d3_q0 = planner.calc_535(M_q0[i], t_current)
                    q1, d_q1, d2_q1, d3_q1 = planner.calc_535(M_q1[i], t_current)
                    q2, d_q2, d2_q2, d3_q2 = planner.calc_535(M_q2[i], t_current)
                    q3, d_q3, d2_q3, d3_q3 = planner.calc_535(M_q3[i], t_current)
                    q4, d_q4, d2_q4, d3_q4 = planner.calc_535(M_q4[i], t_current)
                    q5, d_q5, d2_q5, d3_q5 = planner.calc_535(M_q5[i], t_current)
                    break
            planned_q0.append(q0*57.3), planned_d_q0.append(d_q0*57.3), planned_d2_q0.append(d2_q0*57.3)
            planned_q1.append(q1*57.3), planned_d_q1.append(d_q1*57.3), planned_d2_q1.append(d2_q1*57.3)
            planned_q2.append(q2*57.3), planned_d_q2.append(d_q2*57.3), planned_d2_q2.append(d2_q2*57.3)
            planned_q3.append(q3*57.3), planned_d_q3.append(d_q3*57.3), planned_d2_q3.append(d2_q3*57.3)
            planned_q4.append(q4*57.3), planned_d_q4.append(d_q4*57.3), planned_d2_q4.append(d2_q4*57.3)
            planned_q5.append(q5*57.3), planned_d_q5.append(d_q5*57.3), planned_d2_q5.append(d2_q5*57.3)
            time_range.append(t_current)

        plotter = 1
        if plotter:
            input_q0 = np.array([-52.08, -40.55, -29.56, -22.91]).reshape(4, 1)
            input_q1 = np.array([-69.89, -62.23, -78.51, -105.22]).reshape(4, 1)
            input_q2 = np.array([-78.37, -98.58, -104.31, -64.42]).reshape(4, 1)
            input_q3 = np.array([-100.74, -91.70, -73.82, -89.82]).reshape(4, 1)
            input_q4 = np.array([74.04, 70.23, 67.29, 65.92]).reshape(4, 1)
            input_q5 = np.array([31.40, 42.84, 54.21, 61.29]).reshape(4, 1)
            # ------------pos-------------
            plt.figure(0)
            plt.title('position x')
            plt.plot(time_sequence, input_q0, '*', color='green', label='control point x')
            plt.plot(time_range, planned_q0, label='X position')
            plt.plot(time_range, planned_d_q0, label='x velocity [°/s]')
            # plt.plot(time_range, planned_ax, label='x acceleration [°/s]')
            plt.legend(), plt.grid()
            plt.ylabel('Value [°]')
            plt.xlabel('Time [s]')
            #
            # plt.figure(1)
            # plt.title('position x')
            # plt.plot(time_sequence, input_q1, '*', color='green', label='control point x')
            # plt.plot(time_range, planned_q1, label='X position')
            # plt.plot(time_range, planned_d_q1, label='x velocity [°/s]')
            # # plt.plot(time_range, planned_ax, label='x acceleration [°/s]')
            # plt.legend(), plt.grid()
            # plt.ylabel('Position [°]')
            # plt.xlabel('Time [s]')
            #
            # plt.figure(2)
            # plt.title('position x')
            # plt.plot(time_sequence, input_q2, '*', color='green', label='control point x')
            # plt.plot(time_range, planned_q2, label='X position')
            # plt.plot(time_range, planned_d_q2, label='x velocity [°/s]')
            # # plt.plot(time_range, planned_ax, label='x acceleration [°/s]')
            # plt.legend(), plt.grid()
            # plt.ylabel('Position [°]')
            # plt.xlabel('Time [s]')
            #
            # plt.figure(3)
            # plt.title('position x')
            # plt.plot(time_sequence, input_q3, '*', color='green', label='control point x')
            # plt.plot(time_range, planned_q3, label='X position')
            # plt.plot(time_range, planned_d_q3, label='x velocity [°/s]')
            # # plt.plot(time_range, planned_ax, label='x acceleration [°/s]')
            # plt.legend(), plt.grid()
            # plt.ylabel('Position [°]')
            # plt.xlabel('Time [s]')
            #
            # plt.figure(4)
            # plt.title('position x')
            # plt.plot(time_sequence, input_q4, '*', color='green', label='control point x')
            # plt.plot(time_range, planned_q4, label='X position')
            # plt.plot(time_range, planned_d_q4, label='x velocity [°/s]')
            # # plt.plot(time_range, planned_ax, label='x acceleration [°/s]')
            # plt.legend(), plt.grid()
            # plt.ylabel('Position [°]')
            # plt.xlabel('Time [s]')
            #
            # plt.figure(5)
            # plt.title('position x')
            # plt.plot(time_sequence, input_q5, '*', color='green', label='control point x')
            # plt.plot(time_range, planned_q5, label='X position')
            # plt.plot(time_range, planned_d_q5, label='x velocity [°/s]')
            # # plt.plot(time_range, planned_ax, label='x acceleration [°/s]')
            # plt.legend(), plt.grid()
            # plt.ylabel('Position [°]')
            # plt.xlabel('Time [s]')
            # ----------vel----------
            # ----------acc----------
            plt.figure(27)
            plt.title('Acceleration')
            plt.plot(time_range, planned_d2_q0, label="d2_q0_planned")
            # plt.plot(time_range, planned_d2_q1, label="d2_q1_planned")
            # plt.plot(time_range, planned_d2_q2, label="d2_q2_planned")
            # plt.plot(time_range, planned_d2_q3, label="d2_q3_planned")
            # plt.plot(time_range, planned_d2_q4, label="d2_q4_planned")
            # plt.plot(time_range, planned_d2_q5, label="d2_q5_planned")
            plt.legend()
            plt.grid()
            plt.ylabel('Acceleration [°/s^2]')
            plt.xlabel('Time [sec]')
            plt.show()



