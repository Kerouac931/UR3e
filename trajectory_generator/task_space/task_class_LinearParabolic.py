import numpy as np
import matplotlib.pyplot as plt
import time

class LinearParabolic():
    def __init__(self, time_sequence, tb, control_point):
        self.tb = tb
        # self.time_sequence = time_sequence + self.tb * 0.5
        self.time_sequence = time_sequence
        self.time_interval = []
        self.num_rows = len(control_point[:, 0])
        self.h = np.diff(self.time_sequence)
        self.px = [ix for ix in control_point[:, 0]]
        self.py = [iy for iy in control_point[:, 1]]
        self.pz = [iy for iy in control_point[:, 2]]
        self.vx = np.diff(self.px) / self.h
        self.vy = np.diff(self.py) / self.h 
        self.vz = np.diff(self.pz) / self.h  # len(self.v) = self.num_rows-1 = 3
        self.ax = np.zeros(self.num_rows)  # len(self.a) = self.num_rows = 4
        self.ay = np.zeros(self.num_rows)
        self.az = np.zeros(self.num_rows)
        # calc acc
        for i in range(self.num_rows):
            # if i == 0:
            #     # first element
            #     self.ax[0] = self.vx[0] / (0.5 * self.tb)
            #     self.ay[0] = self.vy[0] / (0.5 * self.tb)
            #     self.az[0] = self.vz[0] / (0.5 * self.tb)
            if i == 0:
                # first element
                self.ax[0] = self.vx[0] / self.tb
                self.ay[0] = self.vy[0] / self.tb
                self.az[0] = self.vz[0] / self.tb
            elif i == self.num_rows - 1:
                # last element, i = 4, a[4]=-v[3]/tb
                self.ax[self.num_rows - 1] = (0 - self.vx[self.num_rows - 2]) / (0.5 * self.tb)
                self.ay[self.num_rows - 1] = (0 - self.vy[self.num_rows - 2]) / (0.5 * self.tb)
                self.az[self.num_rows - 1] = (0 - self.vz[self.num_rows - 2]) / (0.5 * self.tb)
            else:
                self.ax[i] = (self.vx[i] - self.vx[i - 1]) / self.tb
                self.ay[i] = (self.vy[i] - self.vy[i - 1]) / self.tb
                self.az[i] = (self.vz[i] - self.vz[i - 1]) / self.tb

        # calc time_interval
        for i in range(self.num_rows):
            # this loop is to calculate the time interval for time section judgment with t_current
            # len(time_interval=2n=10)
            # in total 9 curve segments, therefore, 10 interval points.
            self.time_interval.append(self.time_sequence[i] - 0.5 * self.tb)
            self.time_interval.append(self.time_sequence[i] + 0.5 * self.tb)

        # calc time_interval
        # for i in range(self.num_rows):
        #     # this loop is to calculate the time interval for time section judgment with t_current
        #     # len(time_interval=2n=10)
        #     # in total 9 curve segments, therefore, 10 interval points.
        #     if i == 0:
        #         self.time_interval.append(self.time_sequence[i])
        #         self.time_interval.append(self.time_sequence[i] + 0.5 * self.tb)
        #
        #     elif i == self.num_rows-1:
        #         self.time_interval.append(self.time_sequence[i] - 0.5 * self.tb)
        #         self.time_interval.append(self.time_sequence[i])
        #
        #     else:
        #         self.time_interval.append(self.time_sequence[i] - 0.5 * self.tb)
        #         self.time_interval.append(self.time_sequence[i] + 0.5 * self.tb)


    def line(self, axis, j, t):
        # s = p0 + v0*(t-t0)
        if axis == 'x':
            line_position = self.vx[j] * (t - self.time_sequence[j]) + self.px[j]
        elif axis == 'y':
            line_position = self.vy[j] * (t - self.time_sequence[j]) + self.py[j]
        elif axis == 'z':
            line_position = self.vz[j] * (t - self.time_sequence[j]) + self.pz[j]
        return line_position

    def co_parabolic(self):
        # s = p0 + v0*(t-t0) + 0.5*acc*(t-t0)**2
        # in total 4 elements needs to be calculated:
        # t0, p0, v0, acc

        # step1: t0
        # t0_para = self.time_sequence - 0.5 * self.tb
        t0_para = []
        for i in range(len(self.time_interval)):  # n=4, segment=3, time_interval_num = 3*2+1=7
            if i % 2 == 0:  # i = 0, 2, 4, 6
                t0_para.append(self.time_interval[i])
        print('t0_para=', t0_para)
        # step2: p0

        p0_para_x, p0_para_y, p0_para_z = self.px[0], self.py[0], self.pz[0]
        for i in range(self.num_rows - 1):
            p0_para_x = np.append(p0_para_x, self.line('x', i, t0_para[i+1]))
            p0_para_y = np.append(p0_para_y, self.line('y', i, t0_para[i+1]))
            p0_para_z = np.append(p0_para_z, self.line('z', i, t0_para[i+1]))
        # print('p0_para_x=', p0_para_x)

        # step3: v0
        v0_para_x = np.insert(self.vx, 0, 0)
        v0_para_y = np.insert(self.vy, 0, 0)
        v0_para_z = np.insert(self.vz, 0, 0)

        # step4: acc=self.ax
        M_px = [t0_para, p0_para_x, v0_para_x, self.ax]
        M_py = [t0_para, p0_para_y, v0_para_y, self.ay]
        M_pz = [t0_para, p0_para_z, v0_para_z, self.az]

        # M_px = np.array(M_px)
        M_px = np.transpose(M_px)  # shape(M_px) = (5, 4)
        M_py = np.transpose(np.array(M_py))
        M_pz = np.transpose(np.array(M_pz))

        return M_px, M_py, M_pz

    def co_line(self):
        # calc co of line-equation: p-p0 = v(t-t0)
        #  M_l is to store t0, p0, v(slope of the line-segment) and acc(acc=0)
        M_lx = self.time_sequence[0:-1], self.px[0:-1], self.vx
        M_ly = self.time_sequence[0:-1], self.py[0:-1], self.vy
        M_lz = self.time_sequence[0:-1], self.pz[0:-1], self.vz
        M_lx = np.transpose(M_lx)
        M_ly = np.transpose(M_ly)
        M_lz = np.transpose(M_lz)
        # in order to plot conveniently, append acc=0 at the end column of M_l
        M_lx = np.hstack((M_lx, np.zeros((self.num_rows-1, 1))))
        M_ly = np.hstack((M_ly, np.zeros((self.num_rows-1, 1))))
        M_lz = np.hstack((M_lz, np.zeros((self.num_rows-1, 1))))
        '''
        M_lx = \n  
        t0      p0      v0      acc0=0  
        t1      p1      v1      acc1=0
        t2      p2      v2      acc2=0
        '''
        return M_lx, M_ly, M_lz

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
        # calc linear and parabolic equations
        t0, p0, v0, a0 = M[i]
        p = p0 + v0*(t-t0) + 0.5*a0*(t-t0)**2
        v = v0 + a0*(t-t0)
        a = a0
        return p, v, a

if __name__ == '__main__':
    pos_x, pos_y, pos_z, planned_px, planned_py, planned_pz = [], [], [], [], [], []
    planned_ax, planned_ay, planned_az, planned_vx, planned_vy, planned_vz = [], [], [], [], [], []
    time_range = []
    # -------------------------- data set 1 -------------------------------
    input_x = np.array([0, 4, -82.06, 301.28, 227.88]).reshape(5, 1) * 0.01
    input_y = np.array([-261.03, -328.32, -276.68, -131.99, -300.54]).reshape(5, 1) * 0.01
    input_z = np.array([0.441, 192.91, 313.66, 206.41, 358.10]).reshape(5, 1) * 0.01
    # -------------------------- data set 2 -------------------------------
    input_x = np.array([5.05, 42.98, 301.28, 277.88]).reshape(4, 1) * 0.001
    input_y = np.array([-260.95, -320.32, -276.68, -300.54]).reshape(4, 1) * 0.001
    input_z = np.array([440.64, 313.66, 317.05, 358.10]).reshape(4, 1) * 0.001
    input_x = np.array([5.05, 42.98, 301.28, 277.88]).reshape(4, 1)
    input_y = np.array([-260.95, -320.32, -276.68, -300.54]).reshape(4, 1)
    input_z = np.array([440.64, 313.66, 317.05, 358.10]).reshape(4, 1)
    rotation_x = [0.094, 0.094, 0.094, 0.094]
    rotation_y = [-3.04, -3.04, -3.04, -3.04]
    rotation_z = [0.706, 0.706, 0.706, 0.706]

    num_rows = np.shape(input_x)[0]
    control_point = np.hstack((input_x, input_y, input_z))
    time_sequence = np.linspace(0, 9, num_rows)
    # t_blend = 0.2*np.diff(time_sequence)[0]   # setting the blend time
    t_blend = 1
    tf = time_sequence[-1]

    planner = LinearParabolic(time_sequence, t_blend, control_point)
    M_lx, M_ly, M_lz = planner.co_line()
    M_px, M_py, M_pz = planner.co_parabolic()
    M_x = planner.co_linear_parabolic(M_lx, M_px)
    M_y = planner.co_linear_parabolic(M_ly, M_py)
    M_z = planner.co_linear_parabolic(M_lz, M_pz)
    time_interval = planner.time_interval
    print('time_interval=', time_interval)
    print('M_x:t0   p0  v0  acc0\n', M_x)


    t_start = time.time()
    t_current = time.time() - t_start
    while t_current < tf:
        t_current = time.time() - t_start
        for i in range(2*num_rows - 1):
            if time_interval[i] <= t_current < time_interval[i+1]:
                px, vx, ax = planner.calc_lp(M_x, i, t_current)
                py, vy, ay = planner.calc_lp(M_y, i, t_current)
                pz, vz, az = planner.calc_lp(M_z, i, t_current)
                break
        planned_px.append(px), planned_vx.append(vx), planned_ax.append(ax)
        planned_py.append(py), planned_vy.append(vy), planned_ay.append(ay)
        planned_pz.append(pz), planned_vz.append(vz), planned_az.append(az)
        time_range.append(t_current)

    plotter = True
    if plotter:
        # ------------pos-------------
        plt.figure(0)
        plt.title('Linear Parabolic Spline')
        plt.plot(time_sequence, input_x, '*', color='green', label='control point')
        plt.plot(time_range, planned_px, label='position [m]')
        plt.plot(time_range, planned_vx, label='velocity [m/s]')
        plt.plot(time_range, planned_ax, label='acceleration [m/s^2]')
        plt.legend(), plt.grid()
        plt.ylabel('Value ')
        plt.xlabel('Time [s]')
        # ------------pos-------------
        # plt.figure(1)
        # plt.title('linear parabolic spline')
        # plt.plot(time_sequence, input_x, '*', color='green', label='control point x')
        # plt.plot(time_range, planned_px, label='x position')
        # plt.plot(time_range, planned_ax, label='x velocity')
        # plt.plot(time_range, planned_ax, label='x acceleration')
        # plt.legend(), plt.grid()
        # plt.ylabel('Position [mm]')
        # plt.xlabel('Time [s]')
        #
        # plt.figure(2)
        # plt.title('position y')
        # plt.plot(time_sequence, input_y, '*', color='green', label='control point x')
        # plt.plot(time_range, planned_py, label='y position [mm]')
        # plt.plot(time_range, planned_vy, label='y velocity [mm/s]')
        # plt.plot(time_range, planned_ay, label='y acceleration [mm/s]')
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
        # plt.figure(2)
        # plt.title('linear parabolic spline: velocity')
        # plt.plot(time_range, planned_ax, label='x velocity')
        # plt.legend(), plt.grid()
        # plt.ylabel('Velocity [mm/s]')
        # plt.xlabel('Time [s]')
        # # ---------- acc ----------
        # plt.figure(3)
        # plt.title('linear parabolic spline: acceleration')
        # plt.plot(time_range, planned_ax, label='x acceleration')
        # plt.legend(), plt.grid()
        # plt.ylabel('Acceleration [mm/s^2]')
        # plt.xlabel('Time [s]')

        plt.show()






















