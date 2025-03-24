import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


class CubicSpline(object):
    def __init__(self, input_time, control_point, v, a, mode):
        # input time is the time of control points
        # v = [v0, vf] = start and end velocity defined by user
        # a = [a0, af] = start and end acc defined by user
        # mode = 'vel' or 'acc'
        self.a0, self.a1, self.a2, self.a3 = None, None, None, None
        self.time_sequence = [it for it in input_time]
        self.h = np.diff(self.time_sequence)
        # n control points
        self.num_rows = len(control_point[:, 0])
        self.co = np.zeros((self.num_rows - 1, 4))
        self.v0, self.vf = v
        self.a0, self.af = a
        self.mode = mode

    # Step:1
    def calculate_M_vel(self, p):
        # M is the matrix stored all knot velocity
        # M is related to position, therefore, there should be Mx,My,Mz
        # using the data in M can calculate all 4 coefficients
        # p=input_pos,   v0, vf = start_velocity and end_velocity defined by users
        A = np.zeros((self.num_rows, self.num_rows))
        B = np.zeros((self.num_rows, 1))
        for i in range(0, self.num_rows):  # if n=5, i = 0, 1, 2, 3, 4
            if i == 0:
                # first row of the matrix
                A[i, i], A[i, i + 1] = 2, 1
                B[i] = 6*((p[i+1]-p[i])/self.h[i]-self.v0)/self.h[i]
            elif i == self.num_rows - 1:
                # last row of the matrix
                A[i, i - 1], A[i, i] = 1, 2
                B[i] = 6*(self.vf - (p[i]-p[i-1])/self.h[i-1])/self.h[i-1]
            else:  # if n=5, i = 1,2,3
                # 1,2,3 row of the matrix_row:(0,1,2,3,4)
                A[i, i - 1], A[i, i], A[i, i + 1] = self.h[i-1]/(self.h[i-1]+self.h[i]), 2, self.h[i]/(self.h[i-1]+self.h[i])
                # A[i, i - 1], A[i, i], A[i, i + 1] = 0.5, 2, 0.5
                B[i] = 6*((p[i+1]-p[i])/self.h[i] - (p[i]-p[i-1])/self.h[i-1])/(self.h[i]+self.h[i-1])
        '''
        A=
        [[2.  1.  0.  0.  0. ]
         [0.5 2.  0.5 0.  0. ]
         [0.  0.5 2.  0.5 0. ]
         [0.  0.  0.5 2.  0.5]
         [0.  0.  0.  1.  2. ]]
        '''
        # print('A=\n', A)
        # print('B=\n', B)
        M = np.linalg.solve(A, B)
        # print('M=\n', M)
        return M

    def calculate_M_acc2(self, p):
        # this func is for user defined acc, but it doesn't work
        # the matrix may be wrong.
        # All knowledge is from: https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation
        A = np.zeros((self.num_rows, self.num_rows))
        B = np.zeros((self.num_rows, 1))
        for i in range(0, self.num_rows):  # if n=5, i = 0, 1, 2, 3, 4
            if i == 0:
                # first row of the matrix
                A[0, 0] = 2
                B[0] = 2 * self.a0
            elif i == self.num_rows - 1:
                # last row of the matrix
                A[-1, -1] = 2
                B[-1] = 2 * self.af
            else:  # if n=5, i = 1,2,3
                # 1,2,3 row of the matrix_row:(0,1,2,3,4)
                A[i, i - 1], A[i, i], A[i, i + 1] = self.h[i-1]/(self.h[i-1]+self.h[i]), 2, self.h[i]/(self.h[i-1]+self.h[i])
                # A[i, i - 1], A[i, i], A[i, i + 1] = 0.5, 2, 0.5
                B[i] = 6*((p[i+1]-p[i])/self.h[i] - (p[i]-p[i-1])/self.h[i-1])/(self.h[i]+self.h[i-1])
        '''
        A=
        [[2.  0.  0.  0.  0. ]
         [0.5 2.  0.5 0.  0. ]
         [0.  0.5 2.  0.5 0. ]
         [0.  0.  0.5 2.  0.5]
         [0.  0.  0.  0.  2. ]]
        '''
        #print('A=\n', A)
        # print('B=\n', B)
        M = np.linalg.solve(A, B)
        # print('M=\n', M)
        return M
    def calculate_M_acc(self, input_pos):
        A = np.zeros((self.num_rows - 2, self.num_rows - 2))
        B = np.zeros((self.num_rows - 2, 1))
        for i in range(0, self.num_rows - 2):  # if n=5, i = 0, 1, 2
            if i == 0:
                A[i, i], A[i, i + 1] = 4, 1
            elif i == self.num_rows - 3:
                A[i, i - 1], A[i, i] = 1, 4
            else:
                A[i, i - 1], A[i, i], A[i, i + 1] = 1, 4, 1
            B[i] = (6 / self.h[i] ** 2) * (input_pos[i] - 2 * input_pos[i + 1] + input_pos[i + 2])
        '''
        A=
        [[4. 1. 0.]
        [1. 4. 1.]
        [0. 1. 4.]]
        '''
        # print('B=\n', B)
        M = np.linalg.solve(A, B)
        M = np.vstack((0, M))
        M = np.vstack((M, 0))
        # print('M=',M)
        return M
    def calculate_coefficient(self, input_pos):
        t_planned, pos_planned, vel_planned, acc_planned = [], [], [], []
        if self.mode == 'vel':
            M = self.calculate_M_vel(input_pos)
        elif self.mode == 'acc':
            M = self.calculate_M_acc(input_pos)

        co = np.zeros(4)
        for i in range(self.num_rows - 1):  # i = 0, 1, 2, 3
            self.a0 = input_pos[i]
            self.a1 = (input_pos[i + 1] - input_pos[i]) / self.h[i] - self.h[i] / 6 * (M[i + 1] + 2 * M[i])
            self.a2 = M[i] / 2
            self.a3 = (M[i + 1] - M[i]) / 6 / self.h[i]
            co_new = np.array([self.a0, self.a1, self.a2, self.a3]).reshape((1, 4))
            co = np.vstack((co, co_new))
        co = np.delete(co, 0, axis=0)
        # print('co=\n', co)
        return co

    def calculate_with_t_current(self, co, t):
        for i in range(self.num_rows - 1):  # 0, 1, 2 ... num_row-2
            if self.time_sequence[i] <= t < self.time_sequence[i + 1]:
                t_index = i
                break
            elif t == self.time_sequence[-1]:
                t_index = self.num_rows - 2
                break
        a0, a1, a2, a3 = co[t_index]
        t0 = self.time_sequence[t_index]
        pos_current = a0 + a1 * (t - t0) + a2 * (t - t0) ** 2 + a3 * (t - t0) ** 3
        vel_current = a1 + 2 * a2 * (t - t0) + 3 * a3 * (t - t0) ** 2
        acc_current = 2 * a2 + 6 * a3 * (t - t0)
        return pos_current, vel_current, acc_current


# run with dynamic time
if __name__ == "__main__":
    # --------------- B spline data set--------------------
    q0, q1, q2, q3, q4, q5 = [], [], [], [], [], []
    d_q0, d_q1, d_q2, d_q3, d_q4, d_q5 = [], [], [], [], [], []
    d2_q0, d2_q1, d2_q2, d2_q3, d2_q4, d2_q5 = [], [], [], [], [], []
    time_range = []
    # -----------------------data set 1---------------------------
    input_q0 = np.array([-48.55, -51.51, -18.14, -26.47]).reshape(4, 1)
    input_q1 = np.array([-73.10, -82.17, -105.02, -96.04]).reshape(4, 1)
    input_q2 = np.array([-146.65, -103.03, -74.75, -76.17]).reshape(4, 1)
    input_q3 = np.array([-34.08, -69.11, -82.27, -85.72]).reshape(4, 1)
    input_q4 = np.array([65.21, 64.49, 65.59, 66.61]).reshape(4, 1)
    input_q5 = np.array([27.09, 31.15, 66.59, 57.49]).reshape(4, 1)
    v0, vf = 0, 0
    a0, af = 0, 0
    time_sequence = np.linspace(0, 3, 4)
    control_point = np.hstack((input_q0, input_q1, input_q2, input_q3, input_q4, input_q5))
    planner = CubicSpline(time_sequence, control_point, [v0, vf], [a0, af], 'vel')
    co_q0 = planner.calculate_coefficient(input_q0)
    co_q1 = planner.calculate_coefficient(input_q1)
    co_q2 = planner.calculate_coefficient(input_q2)
    co_q3 = planner.calculate_coefficient(input_q3)
    co_q4 = planner.calculate_coefficient(input_q4)
    co_q5 = planner.calculate_coefficient(input_q5)
    t_final = time_sequence[-1]
    t_start = time.time()

    while time.time() - t_start < t_final:
        t_current = time.time() - t_start
        res_q0 = planner.calculate_with_t_current(co_q0, t_current)
        res_q1 = planner.calculate_with_t_current(co_q1, t_current)
        res_q2 = planner.calculate_with_t_current(co_q2, t_current)
        res_q3 = planner.calculate_with_t_current(co_q3, t_current)
        res_q4 = planner.calculate_with_t_current(co_q4, t_current)
        res_q5 = planner.calculate_with_t_current(co_q5, t_current)
        # position
        q0.append(res_q0[0]), q1.append(res_q1[0]), q2.append(res_q2[0])
        q3.append(res_q3[0]), q4.append(res_q4[0]), q5.append(res_q5[0])
        # velocity
        d_q0.append(res_q0[1]), d_q1.append(res_q1[1]), d_q2.append(res_q2[1])
        d_q3.append(res_q3[1]), d_q4.append(res_q4[1]), d_q5.append(res_q5[1])
        # acceleration
        d2_q0.append(res_q0[2]), d2_q1.append(res_q1[2]), d2_q2.append(res_q2[2])
        d2_q3.append(res_q3[2]), d2_q4.append(res_q4[2]), d2_q5.append(res_q5[2])
        time_range.append(t_current)
    #print("size_pos_x", len(pos_x)) ---> size_pos_x 507977
    plotter = 1
    if plotter == 1:
        # ------------- position -------------#
        plt.figure(1)
        plt.title('position')
        plt.plot(time_sequence, input_q0, 'o', label='control point q0')
        plt.plot(time_sequence, input_q1, 'o', label='control point q1')
        plt.plot(time_sequence, input_q2, 'o', label='control point q2')
        plt.plot(time_sequence, input_q3, 'o', label='control point q3')
        plt.plot(time_sequence, input_q4, 'o', label='control point q4')
        plt.plot(time_sequence, input_q5, 'o', label='control point q5')

        plt.plot(time_range, q0, label='Q0 position')
        plt.plot(time_range, q1, label='Q1 position')
        plt.plot(time_range, q2, label='Q2 position')
        plt.plot(time_range, q3, label='Q3 position')
        plt.plot(time_range, q4, label='Q4 position')
        plt.plot(time_range, q5, label='Q5 position')

        plt.legend(), plt.grid()
        plt.xlabel('Position [mm]')
        plt.ylabel('Time [s]')

        # # ------------- velocitq1 -------------#
        plt.figure(2)
        plt.title('velocitq1')
        plt.plot(time_range, d_q0, label='Q0 velocitq0')
        plt.plot(time_range, d_q1, label='Q1 velocitq1')
        plt.plot(time_range, d_q2, label='Q2 velocitq2')
        plt.plot(time_range, d_q3, label='Q0 velocitq3')
        plt.plot(time_range, d_q4, label='Q1 velocitq4')
        plt.plot(time_range, d_q5, label='Q2 velocitq5')
        plt.legend(), plt.grid()
        plt.xlabel('Velocitq1 [mm/s]')
        plt.ylabel('Time [s]')

        # ------------- acceleration -------------#
        plt.figure(3)
        plt.title('acceleration')
        plt.plot(time_range, d2_q0, label='Q0 acc')
        plt.plot(time_range, d2_q1, label='Q1 acc')
        plt.plot(time_range, d2_q2, label='Q2 acc')
        plt.legend(), plt.grid()
        plt.xlabel('Acceleration [mm/s^2]')
        plt.ylabel('Time [s]')
        # -------------position 3d-------------#
        # fig = plt.figure(4)
        # plt.title('cubic_spline_q0q1q2')
        # aq0 = fig.add_subplot(111, projection='3d')
        # aq0.plot3D(pos_q0, pos_q1, pos_q2)
        # aq0.set_q0label('Q0 Label')
        # aq0.set_q1label('Q1 Label')
        # aq0.set_q2label('Q2 Label')
        # plt.grid(True)

        plt.show()


