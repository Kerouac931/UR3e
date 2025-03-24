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
        self.q0 = [ix for ix in control_point[:, 0]]
        self.q1 = [iy for iy in control_point[:, 1]]
        self.q2 = [iz for iz in control_point[:, 2]]
        self.q3 = [iu for iu in control_point[:, 3]]
        self.q4 = [iv for iv in control_point[:, 4]]
        self.q5 = [iw for iw in control_point[:, 5]]
        self.v_cp = vel
        self.a_cp = acc
        self.mode = mode

    def two_point_to_four(self):
        # from input_x ---> B points ---> [p0, p1, p2, p3]
        result_q0, result_q1, result_q2, result_q3, result_q4, result_q5 = [], [], [], [], [], []
        r_value_q0, r_value_q1, r_value_q2 = [], [], []  # right side value of the eq
        r_value_q3, r_value_q4, r_value_q5 = [], [], []
        # ------------- generate M matrix with prior knowledge------------
        ''' mode vel '''
        if self.mode == 'vel':  # vel mode
            M = np.zeros((self.num_rows, self.num_rows))
            # this loop is to calculate B points
            for i in range(self.num_rows):
                if i == 0:
                    M[0][0], M[0][1] = 2, 1  # first row, first column
                    r_value_q0.append(3 * self.q0[0] + self.v_cp[0])
                    r_value_q1.append(3 * self.q1[0] + self.v_cp[0])
                    r_value_q2.append(3 * self.q2[0] + self.v_cp[0])
                    r_value_q3.append(3 * self.q3[0] + self.v_cp[0])
                    r_value_q4.append(3 * self.q4[0] + self.v_cp[0])
                    r_value_q5.append(3 * self.q5[0] + self.v_cp[0])
                elif i == self.num_rows - 1:
                    M[-1][-2], M[-1][-1] = 1, 2  # last row, last column
                    r_value_q0.append(3 * self.q0[-1] + self.v_cp[1])
                    r_value_q1.append(3 * self.q1[-1] + self.v_cp[1])
                    r_value_q2.append(3 * self.q2[-1] + self.v_cp[1])
                    r_value_q3.append(3 * self.q3[-1] + self.v_cp[1])
                    r_value_q4.append(3 * self.q4[-1] + self.v_cp[1])
                    r_value_q5.append(3 * self.q5[-1] + self.v_cp[1])
                else:
                    M[i][i - 1], M[i][i], M[i][i + 1] = 1, 4, 1
                    r_value_q0.append(6 * self.q0[i])
                    r_value_q1.append(6 * self.q1[i])
                    r_value_q2.append(6 * self.q2[i])
                    r_value_q3.append(6 * self.q3[i])
                    r_value_q4.append(6 * self.q4[i])
                    r_value_q5.append(6 * self.q5[i])
            B_q0 = np.linalg.solve(M, r_value_q0)
            B_q1 = np.linalg.solve(M, r_value_q1)
            B_q2 = np.linalg.solve(M, r_value_q2)
            B_q3 = np.linalg.solve(M, r_value_q3)
            B_q4 = np.linalg.solve(M, r_value_q4)
            B_q5 = np.linalg.solve(M, r_value_q5)

            # this loop is to calculate [p0, p1, p2, p3]
            for i in range(self.num_rows - 1):
                # ------------- joint 0 -------------
                p0 = B_q0[i]
                p3 = B_q0[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q0.append([self.q0[i], p1, p2, self.q0[i + 1]])
                # ------------- joint 1 -------------
                p0 = B_q1[i]
                p3 = B_q1[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q1.append([self.q1[i], p1, p2, self.q1[i + 1]])
                # ------------- joint 2 -------------
                p0 = B_q2[i]
                p3 = B_q2[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q2.append([self.q2[i], p1, p2, self.q2[i + 1]])
                # ------------- joint 3 -------------
                p0 = B_q3[i]
                p3 = B_q3[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q3.append([self.q3[i], p1, p2, self.q3[i + 1]])
                # ------------- joint 4 -------------
                p0 = B_q4[i]
                p3 = B_q4[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q4.append([self.q4[i], p1, p2, self.q4[i + 1]])
                # ------------- joint 5 -------------
                p0 = B_q5[i]
                p3 = B_q5[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q5.append([self.q5[i], p1, p2, self.q5[i + 1]])
        ''' mode acc '''
        if self.mode == 'acc':
            M = np.zeros((self.num_rows - 2, self.num_rows - 2))
            for i in range(self.num_rows - 2):
                if i == 0:
                    # first row
                    M[0][0], M[0][1] = 4, 1
                    r_value_q0.append(6 * self.q0[1] - self.q0[0])
                    r_value_q1.append(6 * self.q1[1] - self.q1[0])
                    r_value_q2.append(6 * self.q2[1] - self.q2[0])
                    r_value_q3.append(6 * self.q3[1] - self.q3[0])
                    r_value_q4.append(6 * self.q4[1] - self.q4[0])
                    r_value_q5.append(6 * self.q5[1] - self.q5[0])

                elif i == self.num_rows - 3:
                    # last row
                    M[-1][-2], M[-1][-1] = 1, 4
                    r_value_q0.append(6 * self.q0[-2] - self.q0[-1])
                    r_value_q1.append(6 * self.q1[-2] - self.q1[-1])
                    r_value_q2.append(6 * self.q2[-2] - self.q2[-1])
                    r_value_q3.append(6 * self.q3[-2] - self.q3[-1])
                    r_value_q4.append(6 * self.q4[-2] - self.q4[-1])
                    r_value_q5.append(6 * self.q5[-2] - self.q5[-1])
                    break
                else:
                    M[i][i - 1], M[i][i], M[i][i + 1] = 1, 4, 1
                    r_value_q0.append(6 * self.q0[i + 1])
                    r_value_q1.append(6 * self.q1[i + 1])
                    r_value_q2.append(6 * self.q2[i + 1])
                    r_value_q3.append(6 * self.q3[i + 1])
                    r_value_q4.append(6 * self.q4[i + 1])
                    r_value_q5.append(6 * self.q5[i + 1])

            B_q0 = np.linalg.solve(M, r_value_q0)
            B_q0 = np.insert(B_q0, 0, self.q0[0])
            B_q0 = np.append(B_q0, self.q0[-1])
            B_q1 = np.linalg.solve(M, r_value_q1)
            B_q1 = np.insert(B_q1, 0, self.q1[0])
            B_q1 = np.append(B_q1, self.q1[-1])
            B_q2 = np.linalg.solve(M, r_value_q2)
            B_q2 = np.insert(B_q2, 0, self.q2[0])
            B_q2 = np.append(B_q2, self.q2[-1])
            B_q3 = np.linalg.solve(M, r_value_q3)
            B_q3 = np.insert(B_q3, 0, self.q3[0])
            B_q3 = np.append(B_q3, self.q3[-1])
            B_q4 = np.linalg.solve(M, r_value_q4)
            B_q4 = np.insert(B_q4, 0, self.q4[0])
            B_q4 = np.append(B_q4, self.q4[-1])
            B_q5 = np.linalg.solve(M, r_value_q5)
            B_q5 = np.insert(B_q5, 0, self.q5[0])
            B_q5 = np.append(B_q5, self.q5[-1])

            for i in range(self.num_rows - 1):
                # ------------- joint 0 -------------
                p0 = B_q0[i]
                p3 = B_q0[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q0.append([self.q0[i], p1, p2, self.q0[i + 1]])
                # ------------- joint 1 -------------
                p0 = B_q1[i]
                p3 = B_q1[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q1.append([self.q1[i], p1, p2, self.q1[i + 1]])
                # ------------- joint 2 -------------
                p0 = B_q2[i]
                p3 = B_q2[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q2.append([self.q2[i], p1, p2, self.q2[i + 1]])
                # ------------- joint 3 -------------
                p0 = B_q3[i]
                p3 = B_q3[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q3.append([self.q3[i], p1, p2, self.q3[i + 1]])
                # ------------- joint 4 -------------
                p0 = B_q4[i]
                p3 = B_q4[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q4.append([self.q4[i], p1, p2, self.q4[i + 1]])
                # ------------- joint 5 -------------
                p0 = B_q5[i]
                p3 = B_q5[i + 1]
                p1 = 2 / 3 * p0 + 1 / 3 * p3
                p2 = 1 / 3 * p0 + 2 / 3 * p3
                result_q5.append([self.q5[i], p1, p2, self.q5[i + 1]])

        return result_q0, result_q1, result_q2, result_q3, result_q4, result_q5

    def calc_cubic_Bezier(self, i, p, t_c):
        p0, p1, p2, p3 = p
        '''
        it is easy to forget yet important to remember that 
        t is in the range of [0,1]
        this equation: t = (t_c - time_sequence[i]) / width_time_interval
        is to transform t_current to the range of [0,1]
        '''
        t = (t_c - self.time_sequence[i]) / self.time_interval_width[i]
        T_pos = [t**3,   t**2,   t,  1]
        T_vel = [3*t**2, 2*t,    1,  0]
        T_acc = [6*t,    2,      0,  0]
        calc_matrix = [[-p0 + 3*p1 - 3*p2 + p3],
                       [3*p0 - 6*p1 + 3*p2],
                       [-3*p0 + 3*p1],
                       [p0]]
        pos = np.dot(T_pos, calc_matrix)
        vel = np.dot(T_vel, calc_matrix)
        acc = np.dot(T_acc, calc_matrix)
        tf = self.time_sequence[-1]

        factor_crt = (self.num_rows-1)/tf  # corrector

        # vel = vel*(m/tf),     acc = acc*(m/tf)^2
        # These two formula can be calculated in math by inserting t = (t_current-t0)/h
        # h = m/tf
        return pos, vel*factor_crt, acc*factor_crt**2



if __name__ == "__main__":
    # -----------------------initialization---------------------------
    q0, q1, q2, q3, q4, q5 = [], [], [], [], [], []
    d_q0, d_q1, d_q2, d_q3, d_q4, d_q5 = [], [], [], [], [], []
    d2_q0, d2_q1, d2_q2, d2_q3, d2_q4, d2_q5 = [], [], [], [], [], []
    planned_q0, planned_q1, planned_q2, planned_q3, planned_q4, planned_q5 = [], [], [], [], [], []
    planned_d_q0, planned_d_q1, planned_d_q2, planned_d_q3, planned_d_q4, planned_d_q5 = [], [], [], [], [], []
    planned_d2_q0, planned_d2_q1, planned_d2_q2, planned_d2_q3, planned_d2_q4, planned_d2_q5 = [], [], [], [], [], []
    time_range = []
    calc_vel, calc_acc = [], []
    # # -----------------------data set 1---------------------------
    # input_q0 = np.array([4.15, 42.98, -82.06, 301.28, 227.88]).reshape(5, 1)
    # input_q1 = np.array([-261.03, -328.32, -276.68, -131.99, -300.54]).reshape(5, 1)
    # input_q2 = np.array([0.441, 192.91, 313.66, 206.41, 358.10]).reshape(5, 1)
    # input_q3 = np.array([4.15, 42.98, -82.06, 301.28, 227.88]).reshape(5, 1) * 0.001
    # input_q4 = np.array([-261.03, -328.32, -276.68, -131.99, -300.54]).reshape(5, 1) * 0.001
    # input_q5 = np.array([0.441, 192.91, 313.66, 206.41, 358.10]).reshape(5, 1) * 0.001
    # # -----------------------data set 2---------------------------
    input_q0 = np.array([-52.08, -40.55, -29.56, -22.91]).reshape(4, 1) * 0.01745
    input_q1 = np.array([-69.89, -62.23, -78.51, -105.22]).reshape(4, 1) * 0.01745
    input_q2 = np.array([-78.37, -98.58, -104.31, -64.42]).reshape(4, 1) * 0.01745
    input_q3 = np.array([-100.74, -91.70, -73.82, -89.82]).reshape(4, 1) * 0.01745
    input_q4 = np.array([74.04, 70.23, 67.29, 65.92]).reshape(4, 1) * 0.01745
    input_q5 = np.array([31.40, 42.84, 54.21, 61.29]).reshape(4, 1) * 0.01745
    control_point = np.hstack((input_q0, input_q1, input_q2, input_q3, input_q4, input_q5))

    time_sequence = np.linspace(0, 9, np.shape(input_q0)[0])
    t_final = time_sequence[-1]
    input_vx = [0, 0]  # start and end vel for the trajectory
    input_ax = [0, 0]  # start and end acc for the trajectory
    planner = CubicBezier(time_sequence, control_point, input_vx, input_ax, 'vel')
    point_set_q0, point_set_q1, point_set_q2, point_set_q3, point_set_q4, point_set_q5 = planner.two_point_to_four()
    t_start = time.time()
    t_current = 0

    while t_current < t_final:
        time_on = time.time()
        t_prev = t_current
        t_current = time.time() - t_start
        dtime = t_current - t_prev

        for i in range(planner.num_rows-1):  # n=4, i = 0, 1, 2
            if planner.time_interval[i] <= t_current < planner.time_interval[i + 1]:
                t_current = time.time() - t_start
                q0, d_q0, d2_q0 = planner.calc_cubic_Bezier(i, point_set_q0[i], t_current)
                # q1, d_q1, d2_q1 = planner.calc_cubic_Bezier(i, point_set_q1[i], t_current)
                # q2, d_q2, d2_q2 = planner.calc_cubic_Bezier(i, point_set_q2[i], t_current)
                # q3, d_q3, d2_q3 = planner.calc_cubic_Bezier(i, point_set_q3[i], t_current)
                # q4, d_q4, d2_q4 = planner.calc_cubic_Bezier(i, point_set_q4[i], t_current)
                # q5, d_q5, d2_q5 = planner.calc_cubic_Bezier(i, point_set_q5[i], t_current)
                break
        planned_q0.append(q0), planned_d_q0.append(d_q0), planned_d2_q0.append(d2_q0)
        # planned_q1.append(q1), planned_d_q1.append(d_q1), planned_d2_q1.append(d2_q1)
        # planned_q2.append(q2), planned_d_q2.append(d_q2), planned_d2_q2.append(d2_q2)
        # planned_q3.append(q3), planned_d_q3.append(d_q3), planned_d2_q3.append(d2_q3)
        # planned_q4.append(q4), planned_d_q4.append(d_q4), planned_d2_q4.append(d2_q4)
        # planned_q5.append(q5), planned_d_q5.append(d_q5), planned_d2_q5.append(d2_q5)
        time_range.append(t_current)

        # derivative of position and velocity
        if len(planned_q0) <= 2:
            calc_vel.append(np.array([0]))
        else:
            dt = time_range[-1] - time_range[-2]
            ds = planned_q0[-1] - planned_q0[-2]
            calc_vel.append(ds / dt)

        if len(calc_vel) <= 2:
            calc_acc.append(np.array([0]))
        else:
            dt = time_range[-1] - time_range[-2]
            dv = calc_vel[-1] - calc_vel[-2]
            calc_acc.append(dv / dt)
        time.sleep(0.004)


    plotter = 1
    if plotter:
        # ------------pos-------------
        plt.figure(1)
        plt.title('position j0')
        plt.plot(time_sequence, input_q0, '*', color='green', label='control point x')
        plt.plot(time_range, planned_q0, label='X position')
        # plt.plot(time_range, planned_d_q0, label='x velocity [m/s]')
        # plt.plot(time_range, planned_d2_q0, label='x acceleration [m/s]')
        plt.legend(), plt.grid()
        plt.ylabel('Position [°]')
        plt.xlabel('Time [s]')
        #
        # plt.figure(2)
        # plt.title('position j1')
        # plt.plot(time_sequence, input_q1, '*', color='green', label='control point x')
        # plt.plot(time_range, planned_q1, label='y position [°]')
        # plt.plot(time_range, planned_d_q1, label='x velocity [m/s]')
        # plt.plot(time_range, planned_d2_q1, label='x acceleration [m/s]')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [°]')
        # plt.xlabel('Time [s]')
        #
        # plt.figure(3)
        # plt.title('position j2')
        # plt.plot(time_sequence, input_q2, '*', color='green', label='control point z')
        # plt.plot(time_range, planned_q2, label='z position [°]')
        # plt.plot(time_range, planned_d_q2, label='x velocity [m/s]')
        # plt.plot(time_range, planned_d2_q2, label='x acceleration [m/s]')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [°]')
        # plt.xlabel('Time [s]')
        #
        # plt.figure(4)
        # plt.title('position j3')
        # plt.plot(time_sequence, input_q3, '*', color='green', label='control point x')
        # plt.plot(time_range, planned_q3, label='X position')
        # plt.plot(time_range, planned_d_q3, label='x velocity [m/s]')
        # plt.plot(time_range, planned_d2_q3, label='x acceleration [m/s]')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [°]')
        # plt.xlabel('Time [s]')
        #
        # plt.figure(5)
        # plt.title('position j4')
        # plt.plot(time_sequence, input_q4, '*', color='green', label='control point x')
        # plt.plot(time_range, planned_q4, label='y position [°]')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [°]')
        # plt.xlabel('Time [s]')
        #
        # plt.figure(6)
        # plt.title('position j5')
        # plt.plot(time_sequence, input_q5, '*', color='green', label='control point z')
        # plt.plot(time_range, planned_q5, label='z position [°]')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [°]')
        # plt.xlabel('Time [s]')
        #


        # Find max and min for calc_vel
        calc_v_max = np.argmax(calc_vel)
        calc_v_min = np.argmin(calc_vel)
        calc_a_max = np.argmax(calc_acc)
        calc_a_min = np.argmin(calc_acc)
        # Find max and min for planned_d_q0
        planned_v_max = np.argmax(planned_d_q0)
        planned_v_min = np.argmin(planned_d_q0)
        planned_a_max = np.argmax(planned_d2_q0)
        planned_a_min = np.argmin(planned_d2_q0)

        plt.figure(7)
        plt.title('j0_vel')
        plt.plot(time_range, planned_d_q0, linewidth=4, alpha=0.7, label='x velocity [°/s]')
        plt.plot(time_range, calc_vel, label="calc_vel [°/s]")
        # plt.plot(time_range[calc_v_max], calc_vel[calc_v_max], 'o')
        # plt.plot(time_range[calc_v_min], calc_vel[calc_v_min], 'o')
        # plt.plot(time_range[planned_v_max], planned_d_q0[planned_v_max], 'o')
        # plt.plot(time_range[planned_v_min], planned_d_q0[planned_v_min], 'o')
        plt.legend(), plt.grid()
        plt.ylabel('Value')
        plt.xlabel('Time [s]')

        plt.figure(8)
        plt.title('j0_acc')
        plt.plot(time_range, calc_acc, label='calc_acc [°/s^2]')
        plt.plot(time_range, planned_d2_q0, label='x acceleration [°/s^2]')
        plt.plot(time_range[calc_a_max], calc_acc[calc_a_max], 'o')
        plt.plot(time_range[calc_a_min], calc_acc[calc_a_min], 'o')
        plt.plot(time_range[planned_a_max], planned_d2_q0[planned_a_max], 'o')
        plt.plot(time_range[planned_a_min], planned_d2_q0[planned_a_min], 'o')
        plt.legend(), plt.grid()
        plt.ylabel('Value')
        plt.xlabel('Time [s]')

        # -------------- check the result ------------------- #
        print('tf=', time_sequence[-1])
        print('num_seg=', planner.num_rows-1)
        print('vel_max/plan=', calc_vel[calc_v_max]/planned_d_q0[planned_v_max])
        # print('vel_min/plan=', calc_vel[calc_v_min]/planned_d_q0[planned_v_min])
        print('vel_100/plann=', calc_vel[100] / planned_d_q0[100])
        print('vel_200/plann=', calc_vel[200] / planned_d_q0[200])
        print('vel_300/plann=', calc_vel[300] / planned_d_q0[300])
        print('vel_400/plann=', calc_vel[400] / planned_d_q0[400])
        print('vel_500/plann=', calc_vel[500] / planned_d_q0[500])
        print('vel_600/plann=', calc_vel[600] / planned_d_q0[600])
        print('vel_700/plann=', calc_vel[700] / planned_d_q0[700])
        print('vel_800/plann=', calc_vel[800] / planned_d_q0[800])
        print('\n')
        print('acc_100/plann', calc_acc[100] / planned_d2_q0[100])
        print('acc_200/plann', calc_acc[200] / planned_d2_q0[200])
        print('acc_300/plann', calc_acc[300] / planned_d2_q0[300])
        print('acc_400/plann', calc_acc[400] / planned_d2_q0[400])
        print('acc_500/plann', calc_acc[500] / planned_d2_q0[500])
        print('acc_600/plann', calc_acc[600] / planned_d2_q0[600])
        print('acc_700/plann', calc_acc[700] / planned_d2_q0[700])
        print('acc_800/plann', calc_acc[800] / planned_d2_q0[800])

        plt.show()

