import numpy as np
import time
import matplotlib.pyplot as plt

class CubicBezier():

    def __init__(self, time_sequence, control_point, vel, acc, mode):
        # vel: is the start and end vel of the trajectory, it should be vel=[v0, vf]
        # acc: is the start and end acc of the trajectory, it should be acc=[a0, af]
        # mode = 'vel' or 'acc'
        self.time_sequence = time_sequence
        self.time_interval = list(time_sequence)
        self.time_interval_width = np.diff(self.time_sequence)
        self.num_rows = len(control_point[:, 0])
        self.px = [ix for ix in control_point[:, 0]]
        self.py = [iy for iy in control_point[:, 1]]
        self.pz = [iy for iy in control_point[:, 2]]
        self.v_cp = vel
        self.a_cp = acc
        self.mode = mode

    def two_point_to_four(self):
        # from input_x ---> B points ---> [p0, p1, p2, p3]
        result_x, result_y, result_z = [], [], []
        r_value_x, r_value_y, r_value_z = [], [], []  # right side value of the eq
        # ------------- generate M matrix with prior knowledge------------
        ''' mode vel '''
        if self.mode == 'vel':  # vel mode
            M = np.zeros((self.num_rows, self.num_rows))
            # this loop is to calculate B points
            for i in range(self.num_rows):
                if i == 0:
                    M[0][0], M[0][1] = 2, 1  # first row, first column
                    r_value_x.append(3 * self.px[0] + self.v_cp[0])
                    r_value_y.append(3 * self.py[0] + self.v_cp[0])
                    r_value_z.append(3 * self.pz[0] + self.v_cp[0])
                elif i == self.num_rows - 1:
                    M[-1][-2], M[-1][-1] = 1, 2  # last row, last column
                    r_value_x.append(3 * self.px[-1] + self.v_cp[-1])
                    r_value_y.append(3 * self.py[-1] + self.v_cp[-1])
                    r_value_z.append(3 * self.pz[-1] + self.v_cp[-1])
                else:
                    M[i][i - 1] = 1
                    M[i][i] = 4
                    M[i][i + 1] = 1
                    r_value_x.append(6 * self.px[i])
                    r_value_y.append(6 * self.py[i])
                    r_value_z.append(6 * self.pz[i])
            B_x = np.linalg.solve(M, r_value_x)
            B_y = np.linalg.solve(M, r_value_y)
            B_z = np.linalg.solve(M, r_value_z)

            # this loop is to calculate [p0, p1, p2, p3]
            for i in range(self.num_rows-1):
                # ------------- x axis -------------
                p0 = B_x[i]
                p3 = B_x[i+1]
                p1 = 2/3*p0 + 1/3*p3
                p2 = 1/3*p0 + 2/3*p3
                result_x.append([self.px[i], p1, p2, self.px[i+1]])
                # ------------- y axis -------------
                p0 = B_y[i]
                p3 = B_y[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_y.append([self.py[i], p1, p2, self.py[i+1]])
                # ------------- z axis -------------
                p0 = B_z[i]
                p3 = B_z[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_z.append([self.pz[i], p1, p2, self.pz[i+1]])

        ''' mode acc '''
        if self.mode == 'acc':
            M = np.zeros((self.num_rows-2, self.num_rows-2))
            for i in range(self.num_rows-2):
                if i == 0:
                    M[0][0], M[0][1] = 4, 1
                    r_value_x.append(6 * self.px[1] - self.px[0])
                    r_value_y.append(6 * self.py[1] - self.py[0])
                    r_value_z.append(6 * self.pz[1] - self.pz[0])
                elif i == self.num_rows-3:
                    M[-1][-2], M[-1][-1] = 1, 4
                    r_value_x.append(6*self.px[-2]-self.px[-1])
                    r_value_y.append(6*self.py[-2]-self.py[-1])
                    r_value_z.append(6*self.pz[-2]-self.pz[-1])
                    break
                else:
                    M[i][i-1], M[i][i], M[i][i+1] = 1, 4, 1
                    r_value_x.append(6*self.px[i+1])
                    r_value_y.append(6*self.py[i+1])
                    r_value_z.append(6*self.pz[i+1])

            B_x = np.linalg.solve(M, r_value_x)
            B_x = np.insert(B_x, 0, self.px[0])
            B_x = np.append(B_x, self.px[-1])
            print('B_x', B_x)

            B_y = np.linalg.solve(M, r_value_y)
            B_y = np.insert(B_y, 0, self.py[0])
            B_y = np.append(B_y, self.py[-1])

            B_z = np.linalg.solve(M, r_value_z)
            B_z = np.insert(B_z, 0, self.pz[0])
            B_z = np.append(B_z, self.pz[-1])

            for i in range(self.num_rows-1):
                # ------------- x axis -------------
                p0 = B_x[i]
                p3 = B_x[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_x.append([self.px[i], p1, p2, self.px[i + 1]])
                # ------------- y ayis -------------
                p0 = B_y[i]
                p3 = B_y[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_y.append([self.py[i], p1, p2, self.py[i + 1]])
                # ------------- z azis -------------
                p0 = B_z[i]
                p3 = B_z[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_z.append([self.pz[i], p1, p2, self.pz[i + 1]])

        return result_x, result_y, result_z

    def calc_cubic_Bezier(self, i, p, t_c):
        p0, p1, p2, p3 = p
        '''
        it is easy to forget yet important to remember that 
        t is in the range of [0,1]
        this equation: t = (t_c - time_sequence[i]) / width_time_interval
        is to transform t_current to the range of [0,1]
        '''
        t = (t_c - self.time_sequence[i]) / self.time_interval_width[i]
        T_pos = [t**3,   t**2,  t**1,   1]
        T_vel = [3*t**2, 2*t,   1,      0]
        T_acc = [6*t,    2,     0,      0]
        calc_matrix = [[-p0 + 3*p1 - 3*p2 + p3],
                       [3*p0 - 6*p1 + 3*p2],
                       [-3*p0 + 3*p1],
                       [p0]
                       ]
        pos = np.dot(T_pos, calc_matrix)
        vel = np.dot(T_vel, calc_matrix)
        acc = np.dot(T_acc, calc_matrix)
        tf = self.time_sequence[-1]

        # corrector
        factor_crt = (self.num_rows - 1) / tf

        # vel = vel*(m/tf),     acc = acc*(m/tf)^2
        # These two formula can be calculated in math by inserting t = (t_current-t0)/h
        # factor_crt = h = m/tf, m is the number of curve-segments.

        return pos, vel*factor_crt, acc*factor_crt**2

if __name__ == "__main__":
    # -----------------------initialization---------------------------
    pos_x, pos_y, pos_z, v_x, v_y, v_z, a_x, a_y, a_z = [], [], [], [], [], [], [], [], []
    planned_px, planned_py, planned_pz = [], [], []
    planned_ax, planned_ay, planned_az, planned_vx, planned_vy, planned_vz = [], [], [], [], [], []
    time_range = []
    calc_vel, calc_acc = [], []
    # # -----------------------data set 1---------------------------
    # input_x = np.array([4.15, 42.98, -82.06, 301.28, 227.88]).reshape(5, 1)
    # input_y = np.array([-261.03, -328.32, -276.68, -131.99, -300.54]).reshape(5, 1)
    # input_z = np.array([0.441, 192.91, 313.66, 206.41, 358.10]).reshape(5, 1)
    # # -----------------------data set 2---------------------------
    input_x = np.array([5.05, 42.98, 150.15, 277.88]).reshape(4, 1) * 0.001
    input_y = np.array([-260.95, -250.25, -276.68, -300.54]).reshape(4, 1) * 0.001
    input_z = np.array([440.64, 380.38, 317.05, 358.10]).reshape(4, 1) * 0.001
    rotation_x = [0.094, 0.094, 0.094, 0.094]
    rotation_y = [-3.04, -3.04, -3.04, -3.04]
    rotation_z = [0.706, 0.706, 0.706, 0.706]

    # input_x = np.array([5.05, 42.98]).reshape(2, 1) * 0.001
    # input_y = np.array([-260.95, -240]).reshape(2, 1) * 0.001
    # input_z = np.array([440.64, 380.38]).reshape(2, 1) * 0.001

    control_point = np.hstack((input_x, input_y, input_z))
    input_vx = [0, 0]  # start and end vel for the trajectory
    input_ax = [0, 0]  # start and end acc for the trajectory
    time_sequence = np.linspace(0, 9, np.shape(input_x)[0])
    t_final = time_sequence[-1]

    planner = CubicBezier(time_sequence, control_point, input_vx, input_ax, 'acc')
    point_set_x, point_set_y, point_set_z = planner.two_point_to_four()

    t_start = time.time()
    t_current = time.time() - t_start
    while t_current < t_final:
        t_current = time.time() - t_start
        for i in range(planner.num_rows-1):  # n=4, i = 0, 1, 2
            if planner.time_interval[i] <= t_current < planner.time_interval[i + 1]:
                t_current = time.time() - t_start
                px, vx, ax = planner.calc_cubic_Bezier(i, point_set_x[i], t_current)
                py, vy, ay = planner.calc_cubic_Bezier(i, point_set_y[i], t_current)
                pz, vz, az = planner.calc_cubic_Bezier(i, point_set_z[i], t_current)
                break
        planned_px.append(px), planned_vx.append(vx), planned_ax.append(ax)
        planned_py.append(py), planned_vy.append(vy), planned_ay.append(ay)
        planned_pz.append(pz), planned_vz.append(vz), planned_az.append(az)
        time_range.append(t_current)

        # derivative of position and velocity
        if len(planned_px) <= 2:
            calc_vel.append(np.array([0]))
        else:
            dt = time_range[-1] - time_range[-2]
            ds = planned_px[-1] - planned_px[-2]
            calc_vel.append(ds / dt)

        if len(calc_vel) <= 2:
            calc_acc.append(np.array([0]))
        else:
            dt = time_range[-1] - time_range[-2]
            dv = calc_vel[-1] - calc_vel[-2]
            calc_acc.append(dv / dt)
        time.sleep(0.004)


    plotter = True
    if plotter:
        # ------------pos-------------
        # fig = plt.figure(0)
        # plt.title('cubic_B_spline_xyz')
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(input_x[0], input_y[0], input_z[0], color='red', label='start point')
        # ax.scatter(input_x[-1], input_y[-1], input_z[-1], color='blue', label='desired point')
        # ax.scatter(input_x[1:-1], input_y[1:-1], input_z[1:-1], color='green', label='via point')
        # ax.plot3D(planned_px, planned_py, planned_pz, color='orange', label='planned trajectory')
        # ax.plot3D(px, py, pz, color='green', linewidth=0.5, label='actual trajectory')
        # ax.set_xlabel('X position [m]')
        # ax.set_ylabel('Y position [m]')
        # ax.set_zlabel('Z position [m]')
        # ax.legend()
        # plt.grid(True)
        plt.figure(1)
        plt.title('Cubic Bezier Spline')
        plt.plot(time_sequence, input_x, '*', color='green', label='control points')
        plt.plot(time_range, planned_px, label='position [m]')
        plt.plot(time_range, planned_vx, label='velocity [m/s]')
        plt.plot(time_range, planned_ax, label='acceleration [m/s]')
        plt.legend(), plt.grid()
        plt.ylabel('Value')
        plt.xlabel('Time [s]')
        #

        calc_v_max = np.argmax(calc_vel)
        calc_v_min = np.argmin(calc_vel)
        calc_a_max = np.argmax(calc_acc)
        calc_a_min = np.argmin(calc_acc)
        # Find max and min for planned_d_q0
        planned_v_max = np.argmax(planned_vx)
        planned_v_min = np.argmin(planned_vx)
        planned_a_max = np.argmax(planned_ax)
        planned_a_min = np.argmin(planned_ax)

        plt.figure(7)
        plt.title('j0_vel')
        plt.plot(time_range, planned_vx, linewidth=4, alpha=0.7, label='x velocity [°/s]')
        plt.plot(time_range, calc_vel, label="calc_vel [°/s]")
        plt.legend(), plt.grid()
        plt.ylabel('Value')
        plt.xlabel('Time [s]')

        plt.figure(8)
        plt.title('j0_acc')
        plt.plot(time_range, calc_acc, label='calc_acc [°/s^2]')
        plt.plot(time_range, planned_ax, label='x acceleration [°/s^2]')
        plt.legend(), plt.grid()
        plt.ylabel('Value')
        plt.xlabel('Time [s]')

        # -------------- check the result ------------------- #
        print('tf=', time_sequence[-1])
        print('num_seg=', planner.num_rows-1)
        print('vel_max/plan=', calc_vel[calc_v_max] / planned_vx[planned_v_max])
        print('vel_100/plann=', calc_vel[100] / planned_vx[100])
        print('vel_200/plann=', calc_vel[200] / planned_vx[200])
        print('vel_300/plann=', calc_vel[300] / planned_vx[300])
        print('vel_400/plann=', calc_vel[400] / planned_vx[400])
        print('vel_500/plann=', calc_vel[500] / planned_vx[500])
        print('vel_600/plann=', calc_vel[600] / planned_vx[600])
        print('vel_700/plann=', calc_vel[700] / planned_vx[700])
        print('vel_800/plann=', calc_vel[800] / planned_vx[800])
        print('\n')
        print('acc_100/plann', calc_acc[100] / planned_ax[100])
        print('acc_200/plann', calc_acc[200] / planned_ax[200])
        print('acc_300/plann', calc_acc[300] / planned_ax[300])
        print('acc_400/plann', calc_acc[400] / planned_ax[400])
        print('acc_500/plann', calc_acc[500] / planned_ax[500])
        print('acc_500/plann', calc_acc[500] / planned_ax[500])
        print('acc_600/plann', calc_acc[600] / planned_ax[600])
        print('acc_700/plann', calc_acc[700] / planned_ax[700])
        print('acc_800/plann', calc_acc[800] / planned_ax[800])

        plt.show()
