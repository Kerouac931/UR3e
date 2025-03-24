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
        self.px = [ix for ix in control_point[:, 0]]
        self.py = [iy for iy in control_point[:, 1]]
        self.pz = [iy for iy in control_point[:, 2]]
        # n control points
        self.num_rows = len(self.px)
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
        # self.px= [1, -1, 1, 4, 7]
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
    pos_x, pos_y, pos_z, v_x, v_y, v_z, a_x, a_y, a_z = [], [], [], [], [], [], [], [], []
    time_range = []
    # -----------------------data set 1---------------------------
    # input_x = np.array([4.15, 42.98, -82.06, 301.28, 227.88]).reshape(5, 1) * 0.001
    # input_y = np.array([-261.03, -328.32, -276.68, -131.99, -300.54]).reshape(5, 1) * 0.001
    # input_z = np.array([0.441, 192.91, 313.66, 206.41, 358.10]).reshape(5, 1) * 0.001
    # -----------------------data set 2---------------------------
    input_x = np.array([5.05, 42.98, 150.15, 277.88]).reshape(4, 1) * 0.001
    input_y = np.array([-260.95, -250.25, -276.68, -300.54]).reshape(4, 1) * 0.001
    input_z = np.array([440.64, 380.38, 317.05, 358.10]).reshape(4, 1) * 0.001

    v0, vf = 0, 0
    a0, af = 0, 0

    time_sequence = np.linspace(0, 9, 4)
    control_point = np.hstack((input_x, input_y, input_z))
    planner = CubicSpline(time_sequence, control_point, [v0, vf], [a0, af], 'acc')
    co_x = planner.calculate_coefficient(input_x)
    co_y = planner.calculate_coefficient(input_y)
    co_z = planner.calculate_coefficient(input_z)

    t_final = time_sequence[-1]
    t_start = time.time()

    while time.time() - t_start < t_final:
        t_current = time.time() - t_start
        res_x = planner.calculate_with_t_current(co_x, t_current)
        res_y = planner.calculate_with_t_current(co_y, t_current)
        res_z = planner.calculate_with_t_current(co_z, t_current)
        # position
        pos_x.append(res_x[0]), pos_y.append(res_y[0]), pos_z.append(res_z[0])
        # velocity
        v_x.append(res_x[1]), v_y.append(res_y[1]), v_z.append(res_z[1])
        # acceleration
        a_x.append(res_x[2]), a_y.append(res_y[2]), a_z.append(res_z[2])
        time_range.append(t_current)
    #print("size_pos_x", len(pos_x)) ---> size_pos_x 507977
    plotter = 1
    if plotter:
        # ------------- position -------------#
        # plt.figure(0)
        # plt.plot(pos_x, pos_y, label='XZ position')
        # plt.legend()
        # plt.grid()
        # plt.ylabel('x Position [mm]')
        # plt.xlabel('y Position [mm]')

        # plt.figure(1)
        # plt.title('CubicSpline: Position')
        # plt.plot(time_sequence, input_x, 'o', label='control point x')
        # # plt.plot(time_sequence, input_y, 'o', label='control point y')
        # # plt.plot(time_sequence, input_z, 'o', label='control point z')
        # plt.plot(time_range, pos_x, label='X position')
        # # plt.plot(time_range, pos_y, label='Y position')
        # # plt.plot(time_range, pos_z, label='Z position')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [m]')
        # plt.xlabel('Time [s]')
        #
        # # # ------------- velocity -------------#
        # plt.figure(2)
        # plt.title('CubicSpline: Velocity')
        # plt.plot(time_range, v_x, label='X velocity')
        # # plt.plot(time_range, v_y, label='Y velocity')
        # # plt.plot(time_range, v_z, label='Z velocity')
        # plt.legend(), plt.grid()
        # plt.ylabel('Velocity [m/s]')
        # plt.xlabel('Time [s]')
        #
        # # ------------- acceleration -------------#
        # plt.figure(3)
        # plt.title('CubicSpline: Acceleration')
        # plt.plot(time_range, a_x, label='X acc')
        # # plt.plot(time_range, a_y, label='Y acc')
        # # plt.plot(time_range, a_z, label='Z acc')
        # plt.legend(), plt.grid()
        # plt.ylabel('Acceleration [m/s^2]')
        # plt.xlabel('Time [s]')
        #
        # # -------------position 3d-------------#
        # # fig = plt.figure(4)
        # # plt.title('cubic_spline_xyz')
        # # ax = fig.add_subplot(111, projection='3d')
        # # ax.plot3D(pos_x, pos_y, pos_z)
        # # ax.set_xlabel('X Label')
        # # ax.set_ylabel('Y Label')
        # # ax.set_zlabel('Z Label')
        # # plt.grid(True)
        #
        plt.figure(0)
        plt.title('Cubic Spline')
        plt.plot(time_sequence, input_x, 'o', label='control point')
        plt.plot(time_range, pos_x, label='position [m]')
        plt.plot(time_range, v_x, label='velocity [m/s]')
        plt.plot(time_range, a_x, label='acc [m/s^2]')
        # plt.plot(time_range, a_y, label='Y acc')
        # plt.plot(time_range, a_z, label='Z acc')
        plt.legend(), plt.grid()
        plt.ylabel('Value')
        plt.xlabel('Time [s]')

        plt.show()

