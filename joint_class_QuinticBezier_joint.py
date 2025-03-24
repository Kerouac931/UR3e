import numpy as np
import time
import matplotlib.pyplot as plt

class QuinticBezier():
    def __init__(self, time_sequence, control_point, v_cp, a_cp):
        # cp_v means the vel at control point, cp_a mean acc at control point
        self.time_sequence = time_sequence
        self.time_interval = list(time_sequence)
        self.time_interval_width = np.diff(self.time_sequence)[0]
        self.num_rows = len(control_point[:, 0])
        self.v_cp = v_cp
        self.a_cp = a_cp

    def two_point_to_six(self, q, joint_num) -> [list]:
        # to store the result, which is 6 value[p0, ... ,  p5]
        result = []
        k = joint_num
        # --------------- x axis ---------------
        for i in range(len(q) - 1):
            p0 = q[i]
            p5 = q[i + 1]
            p1 = q[i] + self.v_cp[k][i]/5
            p2 = 2*p1 - p0 + self.a_cp[k][i]/20
            p4 = p5 - self.v_cp[k][i+1]/5
            p3 = 2*p4 - p5 + self.a_cp[k][i+1]/20
            result.append([p0, p1, p2, p3, p4, p5])
        return result

    def calc_quintic_Bezier(self, i, p, t_c):
        p0, p1, p2, p3, p4, p5 = p
        '''
        it is easy to forget yet important to remember that 
        t is in the range of [0,1]
        this equation: t = (t_c - time_sequence[i]) / width_time_interval
        is to transform t_current to the range of [0,1]
        '''
        t = (t_c - self.time_sequence[i]) / self.time_interval_width
        T_pos = [t**5,    t**4,    t**3,   t**2, t, 1]
        T_vel = [5*t**4,  4*t**3,  3*t**2, 2*t,  1, 0]
        T_acc = [20*t**3, 12*t**2, 6*t,    2,    0, 0]

        pos_matrix = [-p0 + 5*p1 - 10*p2 + 10*p3 - 5*p4 + p5] +\
                     [5*p0 - 20*p1 + 30*p2 - 20*p3 + 5*p4] +\
                     [-10*p0 + 30*p1 - 30*p2 + 10*p3] +\
                     [10*p0 - 20*p1 + 10*p2] +\
                     [-5*p0 + 5*p1] +\
                     [p0]

        pos = np.dot(T_pos, pos_matrix)
        vel = np.dot(T_vel, pos_matrix)
        acc = np.dot(T_acc, pos_matrix)

        tf = self.time_sequence[-1]
        factor_crt = (self.num_rows - 1) / tf  # corrector

        # vel = vel*(m/tf), acc = acc*(m/tf)^2
        # These two formula can be calculated in math by inserting t = (t_current-t0)/h
        # factor_crt = h = m/tf

        return pos, vel*factor_crt, acc*factor_crt ** 2




if __name__ == "__main__":
    # -----------------------initialization---------------------------
    planned_q0, planned_q1, planned_q2, planned_q3, planned_q4, planned_q5 = [], [], [], [], [], []
    planned_d_q0, planned_d_q1, planned_d_q2, planned_d_q3, planned_d_q4, planned_d_q5 = [], [], [], [], [], []
    planned_d2_q0, planned_d2_q1, planned_d2_q2, planned_d2_q3, planned_d2_q4, planned_d2_q5 = [], [], [], [], [], []
    time_range = []
    calc_vel, calc_acc = [], []
    # -----------------------data set 1---------------------------
    input_q0 = np.array([-52.08, -40.55, -29.56, -22.91]).reshape(4, 1) * 0.01745
    input_q1 = np.array([-69.89, -62.23, -78.51, -105.22]).reshape(4, 1) * 0.01745
    input_q2 = np.array([-78.37, -98.58, -104.31, -64.42]).reshape(4, 1) * 0.01745
    input_q3 = np.array([-100.74, -91.70, -73.82, -89.82]).reshape(4, 1) * 0.01745
    input_q4 = np.array([74.04, 70.23, 67.29, 65.92]).reshape(4, 1) * 0.01745
    input_q5 = np.array([31.40, 42.84, 54.21, 61.29]).reshape(4, 1) * 0.01745

    control_point = np.hstack((input_q0, input_q1, input_q2, input_q3, input_q4, input_q4))
    input_vx = [[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]]

    input_ax = [[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]]

    time_sequence = np.linspace(0, 9, len(input_q0))
    t_final = time_sequence[-1]

    planner = QuinticBezier(time_sequence, control_point, input_vx, input_ax)
    point_set_q0, point_set_q1 = planner.two_point_to_six(input_q0, 0), planner.two_point_to_six(input_q1, 1)
    point_set_q2, point_set_q3 = planner.two_point_to_six(input_q2, 2), planner.two_point_to_six(input_q3, 3)
    point_set_q4, point_set_q5 = planner.two_point_to_six(input_q4, 4), planner.two_point_to_six(input_q5, 5)
    t_start = time.time()
    t_current = time.time() - t_start
    while t_current < t_final:
        t_current = time.time() - t_start
        for i in range(planner.num_rows-1):  # n=4, i = 0, 1, 2
            if planner.time_interval[i] <= t_current < planner.time_interval[i + 1]:
                t_current = time.time() - t_start
                q0, d_q0, d2_q0 = planner.calc_quintic_Bezier(i, point_set_q0[i], t_current)
                q1, d_q1, d2_q1 = planner.calc_quintic_Bezier(i, point_set_q1[i], t_current)
                q2, d_q2, d2_q2 = planner.calc_quintic_Bezier(i, point_set_q2[i], t_current)
                q3, d_q3, d2_q3 = planner.calc_quintic_Bezier(i, point_set_q3[i], t_current)
                q4, d_q4, d2_q4 = planner.calc_quintic_Bezier(i, point_set_q4[i], t_current)
                q5, d_q5, d2_q5 = planner.calc_quintic_Bezier(i, point_set_q5[i], t_current)
                break
        planned_q0.append(q0 * 57.3), planned_d_q0.append(d_q0 * 57.3), planned_d2_q0.append(d2_q0 * 57.3)
        # planned_q1.append(q1 * 57.3), planned_d_q1.append(d_q1 * 57.3), planned_d2_q1.append(d2_q1 * 57.3)
        # planned_q2.append(q2 * 57.3), planned_d_q2.append(d_q2 * 57.3), planned_d2_q2.append(d2_q2 * 57.3)
        # planned_q3.append(q3 * 57.3), planned_d_q3.append(d_q3 * 57.3), planned_d2_q3.append(d2_q3 * 57.3)
        # planned_q4.append(q4 * 57.3), planned_d_q4.append(d_q4 * 57.3), planned_d2_q4.append(d2_q4 * 57.3)
        # planned_q5.append(q5 * 57.3), planned_d_q5.append(d_q5 * 57.3), planned_d2_q5.append(d2_q5 * 57.3)
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
        input_q0 = np.array([-52.08, -40.55, -29.56, -22.91]).reshape(4, 1)
        input_q1 = np.array([-69.89, -62.23, -78.51, -105.22]).reshape(4, 1)
        input_q2 = np.array([-78.37, -98.58, -104.31, -64.42]).reshape(4, 1)
        input_q3 = np.array([-100.74, -91.70, -73.82, -89.82]).reshape(4, 1)
        input_q4 = np.array([74.04, 70.23, 67.29, 65.92]).reshape(4, 1)
        input_q5 = np.array([31.40, 42.84, 54.21, 61.29]).reshape(4, 1)
        # ------------pos-------------
        plt.figure(1)
        plt.title('position q0')
        plt.plot(time_sequence, input_q0, '*', color='green', label='control point q0')
        plt.plot(time_range, planned_q0, label='Q0 position')
        # plt.plot(time_range, planned_d_q0, label='q0 velocitq1 [°/s]')
        # plt.plot(time_range, planned_d2_q0, label='q0 acceleration [°/s]')
        plt.legend(), plt.grid()
        plt.ylabel('Position [°]')
        plt.xlabel('Time [s]')
        #
        # plt.figure(2)
        # plt.title('position q1')
        # plt.plot(time_sequence, input_q1, '*', color='green', label='control point q0')
        # plt.plot(time_range, planned_q1, label='q1 position [°]')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [°]')
        # plt.xlabel('Time [s]')
        #
        # plt.figure(3)
        # plt.title('position q2')
        # plt.plot(time_sequence, input_q2, '*', color='green', label='control point q2')
        # plt.plot(time_range, planned_q2, label='q2 position [°]')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [°]')
        # plt.xlabel('Time [s]')
        #
        # plt.figure(4)
        # plt.title('position q3')
        # plt.plot(time_sequence, input_q3, '*', color='green', label='control point q3')
        # plt.plot(time_range, planned_q3, label='Q3 position')
        # plt.plot(time_range, planned_d_q3, label='q3 velocity [°/s]')
        # plt.plot(time_range, planned_d2_q3, label='q3 acceleration [°/s]')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [°]')
        # plt.xlabel('Time [s]')
        #
        # plt.figure(5)
        # plt.title('position q4')
        # plt.plot(time_sequence, input_q4, '*', color='green', label='control point q3')
        # plt.plot(time_range, planned_q4, label='q4 position [°]')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [°]')
        # plt.xlabel('Time [s]')
        #
        # plt.figure(6)
        # plt.title('position q5')
        # plt.plot(time_sequence, input_q5, '*', color='green', label='control point q5')
        # plt.plot(time_range, planned_q5, label='q5 position [°]')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [°]')
        # plt.xlabel('Time [s]')
        # # # ----------vel----------
        # # plt.figure(7)
        # # plt.title('vel joint 0-1-2-3-4-5')
        # # plt.plot(time_range, planned_d_q0, label='q0 velocity [°/s]')
        # # plt.plot(time_range, planned_d_q1, label='q1 velocity [°/s]')
        # # plt.plot(time_range, planned_d_q2, label='q2 velocity [°/s]')
        # # plt.plot(time_range, planned_d_q3, label='q3 velocity [°/s]')
        # # plt.plot(time_range, planned_d_q4, label='q4 velocity [°/s]')
        # # plt.plot(time_range, planned_d_q5, label='q5 velocity [°/s]')
        # # plt.legend(), plt.grid()
        # # plt.ylabel('Velocity[°/s]')
        # # plt.xlabel('Time [s]')
        #
        # # ----------acc----------
        # plt.figure(8)
        # plt.title('acc joint 0-1-2-3-4-5')
        # plt.plot(time_range, planned_d2_q0, label='q0 acceleration [°/s^2]')
        # # plt.plot(time_range, planned_d2_q1, label='q1 acceleration [°/s^2]')
        # # plt.plot(time_range, planned_d2_q2, label='q2 acceleration [°/s^2]')
        # # plt.plot(time_range, planned_d2_q3, label='q3 acceleration [°/s^2]')
        # # plt.plot(time_range, planned_d2_q4, label='q4 acceleration [°/s^2]')
        # # plt.plot(time_range, planned_d2_q5, label='q5 acceleration [°/s^2]')
        # plt.legend(), plt.grid()
        # plt.ylabel('acceleration [°/s^2]')
        # plt.xlabel('Time [s]')

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
        plt.legend(), plt.grid()
        plt.ylabel('Value')
        plt.xlabel('Time [s]')

        plt.figure(8)
        plt.title('j0_acc')
        plt.plot(time_range, calc_acc, label='calc_acc [°/s^2]')
        plt.plot(time_range, planned_d2_q0, label='x acceleration [°/s^2]')
        plt.legend(), plt.grid()
        plt.ylabel('Value')
        plt.xlabel('Time [s]')

        # -------------- check the result ------------------- #
        print('tf=', time_sequence[-1])
        print('num_seg=', planner.num_rows-1)
        print('vel_max/plan=', calc_vel[calc_v_max]/planned_d_q0[planned_v_max])
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
