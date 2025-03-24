import numpy as np
import matplotlib.pyplot as plt
import time

class LinearParabolic():
    def __init__(self, time_sequence, tb, control_point):
        self.tb = tb
        self.time_sequence = time_sequence
        self.time_interval = []
        self.num_rows = len(control_point[:, 0])
        self.h = np.diff(self.time_sequence)
        self.q0 = [ix for ix in control_point[:, 0]]
        self.q1 = [iy for iy in control_point[:, 1]]
        self.q2 = [iy for iy in control_point[:, 2]]
        self.q3 = [ix for ix in control_point[:, 3]]
        self.q4 = [iy for iy in control_point[:, 4]]
        self.q5 = [iy for iy in control_point[:, 5]]

        self.d_q0 = np.diff(self.q0) / self.h
        self.d_q1 = np.diff(self.q1) / self.h
        self.d_q2 = np.diff(self.q2) / self.h  # len(self.v) = self.num_rows-1 = 4
        self.d_q3 = np.diff(self.q3) / self.h
        self.d_q4 = np.diff(self.q4) / self.h
        self.d_q5 = np.diff(self.q5) / self.h  # len(self.v) = self.num_rows-1 = 4

        self.d2_q0 = np.zeros(self.num_rows)  # len(self.a) = self.num_rows = 5
        self.d2_q1 = np.zeros(self.num_rows)
        self.d2_q2 = np.zeros(self.num_rows)
        self.d2_q3 = np.zeros(self.num_rows)  # len(self.a) = self.num_rows = 5
        self.d2_q4 = np.zeros(self.num_rows)
        self.d2_q5 = np.zeros(self.num_rows)

        # calc acc
        for i in range(self.num_rows):
            if i == 0:
                # first element
                self.d2_q0[0] = self.d_q0[0] / self.tb
                self.d2_q1[0] = self.d_q1[0] / self.tb
                self.d2_q2[0] = self.d_q2[0] / self.tb
                self.d2_q3[0] = self.d_q3[0] / self.tb
                self.d2_q4[0] = self.d_q4[0] / self.tb
                self.d2_q5[0] = self.d_q5[0] / self.tb

            elif i == self.num_rows - 1:
                # last element, i = 4, a[4]=-v[3]/tb
                self.d2_q0[self.num_rows - 1] = (0 - self.d_q0[self.num_rows - 2]) / (0.5*self.tb)
                self.d2_q1[self.num_rows - 1] = (0 - self.d_q1[self.num_rows - 2]) / (0.5*self.tb)
                self.d2_q2[self.num_rows - 1] = (0 - self.d_q2[self.num_rows - 2]) / (0.5*self.tb)
                self.d2_q3[self.num_rows - 1] = (0 - self.d_q3[self.num_rows - 2]) / (0.5*self.tb)
                self.d2_q4[self.num_rows - 1] = (0 - self.d_q4[self.num_rows - 2]) / (0.5*self.tb)
                self.d2_q5[self.num_rows - 1] = (0 - self.d_q5[self.num_rows - 2]) / (0.5*self.tb)

            else:
                self.d2_q0[i] = (self.d_q0[i] - self.d_q0[i - 1]) / self.tb
                self.d2_q1[i] = (self.d_q1[i] - self.d_q1[i - 1]) / self.tb
                self.d2_q2[i] = (self.d_q2[i] - self.d_q2[i - 1]) / self.tb
                self.d2_q3[i] = (self.d_q3[i] - self.d_q3[i - 1]) / self.tb
                self.d2_q4[i] = (self.d_q4[i] - self.d_q4[i - 1]) / self.tb
                self.d2_q5[i] = (self.d_q5[i] - self.d_q5[i - 1]) / self.tb

        # calc time_interval
        for i in range(self.num_rows):
            # this loop is to calculate the time interval for time section judgment with t_current
            # len(time_interval=2n=10)
            # in total 9 curve segments, therefore, 10 interval points.
            self.time_interval.append(self.time_sequence[i] - 0.5 * self.tb)
            self.time_interval.append(self.time_sequence[i] + 0.5 * self.tb)

    def line(self, k, j, t):
        if k == 'q0':
            l = self.d_q0[j] * (t - self.time_sequence[j]) + self.q0[j]
        elif k == 'q1':
            l = self.d_q1[j] * (t - self.time_sequence[j]) + self.q1[j]
        elif k == 'q2':
            l = self.d_q2[j] * (t - self.time_sequence[j]) + self.q2[j]
        elif k == 'q3':
            l = self.d_q3[j] * (t - self.time_sequence[j]) + self.q3[j]
        elif k == 'q4':
            l = self.d_q4[j] * (t - self.time_sequence[j]) + self.q4[j]
        elif k == 'q5':
            l = self.d_q5[j] * (t - self.time_sequence[j]) + self.q5[j]
        return l

    def co_parabolic(self):
        # s = p0 + v0*(t-t0) + 0.5*a*(t-t0)**2
        # in total 4 elements needs to be calculated:
        # t0, p0, v0, a
        # step1: t0
        t0_para = self.time_sequence - 0.5 * self.tb
        # step2: p0
        p0_para_q0, p0_para_q1, p0_para_q2 = self.q0[0], self.q1[0], self.q2[0]
        p0_para_q3, p0_para_q4, p0_para_q5 = self.q3[0], self.q4[0], self.q5[0]
        for i in range(self.num_rows - 1):
            p0_para_q0 = np.append(p0_para_q0, self.line('q0', i, t0_para[i+1]))
            p0_para_q1 = np.append(p0_para_q1, self.line('q1', i, t0_para[i+1]))
            p0_para_q2 = np.append(p0_para_q2, self.line('q2', i, t0_para[i+1]))
            p0_para_q3 = np.append(p0_para_q3, self.line('q3', i, t0_para[i+1]))
            p0_para_q4 = np.append(p0_para_q4, self.line('q4', i, t0_para[i+1]))
            p0_para_q5 = np.append(p0_para_q5, self.line('q5', i, t0_para[i+1]))

        # print('p0_para_q0=', p0_para_q0)
        # step3: v0
        v0_para_q0 = np.insert(self.d_q0, 0, 0)
        v0_para_q1 = np.insert(self.d_q1, 0, 0)
        v0_para_q2 = np.insert(self.d_q2, 0, 0)
        v0_para_q3 = np.insert(self.d_q3, 0, 0)
        v0_para_q4 = np.insert(self.d_q4, 0, 0)
        v0_para_q5 = np.insert(self.d_q5, 0, 0)

        # step4: a=self.aq0
        M_q0 = [t0_para, p0_para_q0, v0_para_q0, self.d2_q0]
        M_q1 = [t0_para, p0_para_q1, v0_para_q1, self.d2_q1]
        M_q2 = [t0_para, p0_para_q2, v0_para_q2, self.d2_q2]
        M_q3 = [t0_para, p0_para_q3, v0_para_q3, self.d2_q3]
        M_q4 = [t0_para, p0_para_q4, v0_para_q4, self.d2_q4]
        M_q5 = [t0_para, p0_para_q5, v0_para_q5, self.d2_q5]

        # M_q0 = np.arrange(M_q0)
        M_q0 = np.transpose(M_q0)  # shape(M_q0) = (5, 4)
        M_q1 = np.transpose(np.array(M_q1))
        M_q2 = np.transpose(np.array(M_q2))
        M_q3 = np.transpose(M_q3)  # shape(M_q3) = (5, 4)
        M_q4 = np.transpose(np.array(M_q4))
        M_q5 = np.transpose(np.array(M_q5))


        return M_q0, M_q1, M_q2, M_q3, M_q4, M_q5

    def co_line(self):
        # calc co of line-equation
        M_lq0 = self.time_sequence[0:-1], self.q0[0:-1], self.d_q0
        M_lq1 = self.time_sequence[0:-1], self.q1[0:-1], self.d_q1
        M_lq2 = self.time_sequence[0:-1], self.q2[0:-1], self.d_q2
        M_lq3 = self.time_sequence[0:-1], self.q3[0:-1], self.d_q3
        M_lq4 = self.time_sequence[0:-1], self.q4[0:-1], self.d_q4
        M_lq5 = self.time_sequence[0:-1], self.q5[0:-1], self.d_q5
        M_lq0 = np.transpose(M_lq0)
        M_lq1 = np.transpose(M_lq1)
        M_lq2 = np.transpose(M_lq2)
        M_lq3 = np.transpose(M_lq3)
        M_lq4 = np.transpose(M_lq4)
        M_lq5 = np.transpose(M_lq5)
        # in order to plot conveniently, append a 0 at the end of M_l
        M_lq0 = np.hstack((M_lq0, np.zeros((self.num_rows - 1, 1))))
        M_lq1 = np.hstack((M_lq1, np.zeros((self.num_rows - 1, 1))))
        M_lq2 = np.hstack((M_lq2, np.zeros((self.num_rows - 1, 1))))
        M_lq3 = np.hstack((M_lq3, np.zeros((self.num_rows - 1, 1))))
        M_lq4 = np.hstack((M_lq4, np.zeros((self.num_rows - 1, 1))))
        M_lq5 = np.hstack((M_lq5, np.zeros((self.num_rows - 1, 1))))

        return M_lq0, M_lq1, M_lq2, M_lq3, M_lq4, M_lq5

    def co_linear_parabolic(self, M_l, M_p):
        # combine co of linear and parabolic equation, in order to plot conveniently
        M = []
        for i in range(self.num_rows):
            if i == self.num_rows-1:
                M.append(M_p[-1])
            else:
                M.append(M_p[i])
                M.append(M_l[i])
        M = np.array(M).reshape(2*self.num_rows-1, 4)

        return M

    def calc_lp(self, M, i, t):
        t0, p0, v0, a0 = M[i]
        p = p0 + v0*(t-t0) + 0.5*a0*(t-t0)**2
        v = v0 + a0*(t-t0)
        a = a0
        return p, v, a

if __name__ == '__main__':
    planned_q0, planned_q1, planned_q2, planned_q3, planned_q4, planned_q5 = [], [], [], [], [], []
    planned_d2_q0, planned_d2_q1, planned_d2_q2, planned_d2_q3, planned_d2_q4, planned_d2_q5 = [], [], [], [], [], []
    planned_d_q0, planned_d_q1, planned_d_q2, planned_d_q3, planned_d_q4, planned_d_q5 = [], [], [], [], [], []
    time_range = []
    # ------------------------data set 1-----------------------------
    # input_q0 = np.array([4.15, 42.98, 301.28, 227.88]).reshape(4, 1)
    # input_q1 = np.array([-261.03, -328.32, -276.68, -300.54]).reshape(4, 1)
    # input_q2 = np.array([192.91, 313.66, 206.41, 358.10]).reshape(4, 1)
    # input_q3 = np.array([4.15, 42.98, 301.28, 227.88]).reshape(4, 1)
    # input_q4 = np.array([-261.03, -328.32, -276.68, -300.54]).reshape(4, 1)
    # input_q5 = np.array([192.91, 313.66, 206.41, 358.10]).reshape(4, 1)
    # ------------------------data set 2-----------------------------
    input_q0 = np.array([-0.8471975, -0.8988495, -0.316543, -0.4619015]).reshape(4, 1)
    input_q1 = np.array([-1.275595, -1.4338665, -1.832599, -1.675898]).reshape(4, 1)
    input_q2 = np.array([-2.5590425, -1.7978735, -1.3043875, -1.3291665]).reshape(4, 1)
    input_q3 = np.array([-0.594696, -1.2059695, -1.4356115, -1.495814]).reshape(4, 1)
    input_q4 = np.array([1.1379145, 1.1253505, 1.1445455, 1.1623445]).reshape(4, 1)
    input_q5 = np.array([0.4727205, 0.5435675, 1.1619955, 1.0032005]).reshape(4, 1)
    num_rows = np.shape(input_q0)[0]
    control_point = np.hstack((input_q0, input_q1, input_q2, input_q3, input_q4, input_q5))
    # time for parabolic segments
    time_sequence = np.linspace(0, 9, 4)
    t_blend = 0.2*np.diff(time_sequence)[0]
    # time_sequence = time_sequence + 0.5 * t_blend
    tf = time_sequence[-1]
    planner = LinearParabolic(time_sequence, t_blend, control_point)
    M_lq0, M_lq1, M_lq2, M_lq3, M_lq4, M_lq5 = planner.co_line()
    M_pq0, M_pq1, M_pq2, M_pq3, M_pq4, M_pq5 = planner.co_parabolic()
    M_q0 = planner.co_linear_parabolic(M_lq0, M_pq0)
    M_q1 = planner.co_linear_parabolic(M_lq1, M_pq1)
    M_q2 = planner.co_linear_parabolic(M_lq2, M_pq2)
    M_q3 = planner.co_linear_parabolic(M_lq3, M_pq3)
    M_q4 = planner.co_linear_parabolic(M_lq4, M_pq4)
    M_q5 = planner.co_linear_parabolic(M_lq5, M_pq5)
    # print('M_q0\n', M_q0)
    # print('M_lq0\n', M_lq0)
    # print('M_q0\n', M_q0)
    time_interval = planner.time_interval
    # print("time_interval\n", time_interval)

    t_start = time.time()
    t_current = time.time() - t_start
    count = 0
    while t_current < tf:
        t_current = time.time() - t_start
        for i in range(2*num_rows - 1):
            if time_interval[i] <= t_current < time_interval[i+1]:
                q0, d_q0, d2_q0 = planner.calc_lp(M_q0, i, t_current)
                q1, d_q1, d2_q1 = planner.calc_lp(M_q1, i, t_current)
                q2, d_q2, d2_q2 = planner.calc_lp(M_q2, i, t_current)
                q3, d_q3, d2_q3 = planner.calc_lp(M_q3, i, t_current)
                q4, d_q4, d2_q4 = planner.calc_lp(M_q4, i, t_current)
                q5, d_q5, d2_q5 = planner.calc_lp(M_q5, i, t_current)
                break
        planned_q0.append(q0), planned_d_q0.append(d_q0), planned_d2_q0.append(d2_q0)
        planned_q1.append(q1), planned_d_q1.append(d_q1), planned_d2_q1.append(d2_q1)
        planned_q2.append(q2), planned_d_q2.append(d_q2), planned_d2_q2.append(d2_q2)
        planned_q3.append(q3), planned_d_q3.append(d_q3), planned_d2_q3.append(d2_q3)
        planned_q4.append(q4), planned_d_q4.append(d_q4), planned_d2_q4.append(d2_q4)
        planned_q5.append(q5), planned_d_q5.append(d_q5), planned_d2_q5.append(d2_q5)
        time_range.append(t_current)

    plotter = 1
    if plotter:
        # ------------pos-------------
        plt.figure(1)
        plt.title('position q0')
        plt.plot(time_sequence, input_q0, '*', color='green', label='control point q0')
        plt.plot(time_range, planned_q0, label='Q0 position')
        plt.plot(time_range, planned_d_q0, label='q0 velocity [mm/s]')
        plt.plot(time_range, planned_d2_q0, label='q0 acceleration [mm/s]')
        plt.legend(), plt.grid()
        plt.xlabel('Position [mm]')
        plt.ylabel('Time [s]')

        plt.figure(2)
        plt.title('position q1')
        plt.plot(time_sequence, input_q1, '*', color='green', label='control point q0')
        plt.plot(time_range, planned_q1, label='q1 position [mm]')
        plt.legend(), plt.grid()
        plt.xlabel('Position [mm]')
        plt.ylabel('Time [s]')

        plt.figure(3)
        plt.title('position q2')
        plt.plot(time_sequence, input_q2, '*', color='green', label='control point q2')
        plt.plot(time_range, planned_q2, label='q2 position [mm]')
        plt.legend(), plt.grid()
        plt.xlabel('Position [mm]')
        plt.ylabel('Time [s]')

        plt.figure(4)
        plt.title('position q3')
        plt.plot(time_sequence, input_q3, '*', color='green', label='control point q3')
        plt.plot(time_range, planned_q3, label='Q3 position')
        plt.plot(time_range, planned_d_q3, label='q3 velocity [°/s]')
        plt.plot(time_range, planned_d2_q3, label='q3 acceleration [°/s]')
        plt.legend(), plt.grid()
        plt.xlabel('Position [°]')
        plt.ylabel('Time [s]')

        plt.figure(5)
        plt.title('position q4')
        plt.plot(time_sequence, input_q4, '*', color='green', label='control point q3')
        plt.plot(time_range, planned_q4, label='q4 position [°]')
        plt.legend(), plt.grid()
        plt.xlabel('Position [°]')
        plt.ylabel('Time [s]')

        plt.figure(6)
        plt.title('position q5')
        plt.plot(time_sequence, input_q5, '*', color='green', label='control point q5')
        plt.plot(time_range, planned_q5, label='q5 position [°]')
        plt.legend(), plt.grid()
        plt.xlabel('Position [°]')
        plt.ylabel('Time [s]')

        # ----------vel----------
        plt.figure(7)
        plt.title('velocity joint 0,1,2,3,4,5')
        plt.plot(time_range, planned_d_q0, label='q0 velocity [°/s]')
        plt.plot(time_range, planned_d_q1, label='q1 velocity [°/s]')
        plt.plot(time_range, planned_d_q2, label='q2 velocity [°/s]')
        plt.plot(time_range, planned_d_q3, label='q3 velocity [°/s]')
        plt.plot(time_range, planned_d_q4, label='q4 velocity [°/s]')
        plt.plot(time_range, planned_d_q5, label='q5 velocity [°/s]')
        plt.legend(), plt.grid()

        plt.show()
