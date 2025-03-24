"""
Code written based on servoj example from: https://github.com/davizinho5/RTDE_control_example
"""
import os
import sys
import numpy as np
sys.path.append('')
import logging
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

import time
from matplotlib import pyplot as plt
from task_class_CubicSpline import CubicSpline
from task_class_CubicBezier import CubicBezier

from task_class_LinearParabolic import LinearParabolic
from task_class_Spline535 import Spline535
from task_class_QuinticBezier import QuinticBezier

# -------- functions -------------
def setp_to_list(setp):
    temp = []
    for i in range(0, 6):
        temp.append(setp.__dict__["input_double_register_%i" % i])
    return temp

def list_to_setp(setp, list):
    for i in range(0, 6):
        setp.__dict__["input_double_register_%i" % i] = list[i]
    return setp


# ------------- initialize robot communication stuff -----------------
ROBOT_HOST = '192.168.56.101'
ROBOT_PORT = 30004
FREQUENCY = 500  # send data in 500 Hz instead of default 125Hz
config_filename = 'control_loop_configuration.xml'  # specify xml file for data synchronization
logging.getLogger().setLevel(logging.INFO)
conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe('state')  # Define recipe for access to robot output ex. joints,tcp etc.
setp_names, setp_types = conf.get_recipe('setp')  # Define recipe for access to robot input
watchdog_names, watchdog_types = conf.get_recipe('watchdog')
# -------------------- Establish connection --------------------
con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
connection_state = con.connect()
# check if connection has been established
while connection_state != 0:
    time.sleep(0.5)
    connection_state = con.connect()
print("---------------Successfully connected to the robot-------------\n")
# get controller version
con.get_controller_version()
# ------------------- setup recipes ----------------------------
con.send_output_setup(state_names, state_types, FREQUENCY)
setp = con.send_input_setup(setp_names, setp_types)
# Configure an input package that the external application will send to the robot controller
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0
setp.input_bit_registers0_to_31 = 0
watchdog.input_int_register_0 = 0

# start data synchronization
if not con.send_start():
    sys.exit()
# -------------------data from Sara------------------------
start_pose = [0.00505, -0.261, 0.441, 0.094, -3.04, 0.706]
desired_pose = [0.27788, -0.30054, 0.35810, 0.094, -3.04, 0.706]
# -----------------------data set 1---------------------------
input_x = np.array([5.05, 42.98, 150.15, 277.88]).reshape(4, 1) * 0.001
input_y = np.array([-260.95, -250.25, -276.68, -300.54]).reshape(4, 1) * 0.001
input_z = np.array([440.64, 380.38, 317.05, 358.10]).reshape(4, 1) * 0.001
# input_x = np.array([5.05, 42.98, 301.28, 277.88]).reshape(4, 1) * 0.001
# input_y = np.array([-260.95, -320.32, -276.68, -300.54]).reshape(4, 1) * 0.001
# input_z = np.array([440.64, 313.66, 317.05, 358.10]).reshape(4, 1) * 0.001
rotation_x = [0.094, 0.094, 0.094, 0.094]
rotation_y = [-3.04, -3.04, -3.04, -3.04]
rotation_z = [0.706, 0.706, 0.706, 0.706]
rotation_const = [rotation_x[0], rotation_y[0], rotation_z[0]]
time_sequence = np.linspace(0, 9, len(input_x))                                 # Set the final time
t_final = time_sequence[-1]  # final time of the trajectory
voltage_const = 48  # 48v
# -------------------------------------
state = con.receive()
tcp1 = state.actual_TCP_pose
print(tcp1)
# ------------  mode = 1 (Connection) -----------
while True:
    print('Boolean 1 is False, please click CONTINUE on the Polyscope')
    state = con.receive()
    con.send(watchdog)
    # print(f"runtime state is {state.runtime_state}")
    if state.output_bit_registers0_to_31 == True:
        print('Boolean 1 is True, Robot Program can proceed to mode 1\n')
        break
# print("-------Executing moveJ -----------\n")
#
# watchdog.input_int_register_0 = 1
# con.send(watchdog)  # sending mode == 1
# list_to_setp(setp, start_pose)  # changing initial pose to setp
# con.send(setp)  # sending initial pose
#
# while True:
#     print('Waiting for movej() to finish')
#     state = con.receive()
#     con.send(watchdog)
#     if state.output_bit_registers0_to_31 == False:
#         print('Proceeding to mode 2\n')
#         break

print("-------Executing servoJ  -----------\n")
watchdog.input_int_register_0 = 1
con.send(watchdog)  # sending mode == 2
dt = 1 / FREQUENCY  # 500 Hz    # frequency
plotter = True
# initialization of plot
if plotter:
    time_plot = []
    min_jerk_x, min_jerk_y, min_jerk_z = [], [], []
    min_jerk_vx, min_jerk_vy, min_jerk_vz = [], [], []
    min_jerk_ax, min_jerk_ay, min_jerk_az = [], [], []
    px, py, pz, vx, vy, vz, force_x, force_y, force_z = [], [], [], [], [], [], [], [], []
    planned_px, planned_py, planned_pz = [], [], []
    planned_vx, planned_vy, planned_vz = [], [], []
    planned_ax, planned_ay, planned_az = [], [], []
    joint_current_0, joint_current_1, joint_current_2 = [], [], []
    joint_current_3, joint_current_4, joint_current_5 = [], [], []
    joint_power_0, joint_power_1, joint_power_2 = [], [], []
    joint_power_3, joint_power_4, joint_power_5 = [], [], []
    joint_energy = np.zeros(6)
    time_range = []
    calc_vel_x, calc_vel_y, calc_vel_z = [], [], []
    calc_acc_x, calc_acc_y, calc_acc_z = [], [], []

# ---------------- choose algorithm ----------------
#                 0             1                 2               3               4
mode_list = ['CubicSpline', 'CubicBezier', 'LinearParabolic', 'Spline535', 'QuinticBezier']
# run_mode[]: it matches to mode_list, choose different trajectory algorithm
run_mode = mode_list[4]

if run_mode == 'CubicSpline':
    # ------------------ Control loop initialization -------------------------
    control_point = np.hstack((input_x, input_y, input_z))
    # v0 and vf is where users define the start and end velocity
    v0, vf = 0, 0
    a0, af = 0, 0
    planner = CubicSpline(time_sequence, control_point, [v0, vf], [a0, af], 'vel')
    # res_cubic_spline = cubic_spline(time_sequence, control_point)
    # calc the coefficients of every segment of curve for xyz axis
    co_x = planner.calculate_coefficient(input_x)
    co_y = planner.calculate_coefficient(input_y)
    co_z = planner.calculate_coefficient(input_z)
    # -------------------------Control loop --------------------
    state = con.receive()
    tcp = state.actual_TCP_pose
    t_current = 0
    t_start = state.timestamp
    print('t_start=', t_start)
    while state.timestamp - t_start < t_final:
        # t_init = state.timestamp
        state = con.receive()
        t_prev = t_current
        t_current = state.timestamp - t_start
        dt = t_current - t_prev
        print(f"dt:{dt}")
        # read state from the robot
        if state.runtime_state > 1:
            #   ----------- minimum_jerk trajectory --------------
            if t_current <= t_final:
                print('t_c=', t_current)
                # [position_ref, lin_vel_ref, acceleration_ref] = planner.trajectory_planning(t_current)
                res_x = planner.calculate_with_t_current(co_x, t_current)
                res_y = planner.calculate_with_t_current(co_y, t_current)
                res_z = planner.calculate_with_t_current(co_z, t_current)
                position_ref = [res_x[0], res_y[0], res_z[0]]
                velocity_ref = [res_x[1], res_y[1], res_z[1]]
                acceleration_ref = [res_x[2], res_y[2], res_z[2]]
            # ------------------ impedance -----------------------
            # state.actual_TCP_pose
            # state.actual_TCP_speed
            pose = position_ref + rotation_const  # 6D vector of position given to polsycope
            list_to_setp(setp, pose)
            con.send(setp)

            if plotter:
                time_plot.append(t_current)
                min_jerk_x.append(position_ref[0])
                min_jerk_y.append(position_ref[1])
                min_jerk_z.append(position_ref[2])

                min_jerk_vx.append(velocity_ref[0])
                min_jerk_vy.append(velocity_ref[1])
                min_jerk_vz.append(velocity_ref[2])

                min_jerk_ax.append(acceleration_ref[0])
                min_jerk_ay.append(acceleration_ref[1])
                min_jerk_az.append(acceleration_ref[2])

                px.append(state.actual_TCP_pose[0])
                py.append(state.actual_TCP_pose[1])
                pz.append(state.actual_TCP_pose[2])

                vx.append(state.actual_TCP_speed[0])
                vy.append(state.actual_TCP_speed[1])
                vz.append(state.actual_TCP_speed[2])

                if len(time_plot) <= 10:
                    # calc_vel_x.append(0)
                    # calc_vel_y.append(0)
                    # calc_vel_z.append(0)
                    calc_acc_x.append(0)
                    calc_acc_y.append(0)
                    calc_acc_z.append(0)
                else:
                    # dx = px[-1] - px[-2]
                    # dy = py[-1] - py[-2]
                    # dz = pz[-1] - pz[-2]
                    #
                    # calc_vel_x.append(dx/dt)
                    # calc_vel_y.append(dy/dt)
                    # calc_vel_z.append(dz/dt
                    dvx = vx[-1] - vx[-3]
                    dvy = vy[-1] - vy[-3]
                    dvz = vz[-1] - vz[-3]
                    calc_acc_x.append(dvx/(dt*2))
                    calc_acc_y.append(dvy/(dt*2))
                    calc_acc_z.append(dvz/(dt*2))

                joint_current_0.append(state.actual_current[0])
                joint_current_1.append(state.actual_current[1])
                joint_current_2.append(state.actual_current[2])
                joint_current_3.append(state.actual_current[3])
                joint_current_4.append(state.actual_current[4])
                joint_current_5.append(state.actual_current[5])

                # joint power: j_p
                j_p = np.array(state.actual_current) * voltage_const
                joint_power_0.append(j_p[0])
                joint_power_1.append(j_p[1])
                joint_power_2.append(j_p[2])
                joint_power_3.append(j_p[3])
                joint_power_4.append(j_p[4])
                joint_power_5.append(j_p[5])

                # accumulation of consumed energy
                # j_e: joint_energy
                j_e = abs(j_p) * dt
                joint_energy[0] += j_e[0]
                joint_energy[1] += j_e[1]
                joint_energy[2] += j_e[2]
                joint_energy[3] += j_e[3]
                joint_energy[4] += j_e[4]
                joint_energy[5] += j_e[5]

elif run_mode == 'CubicBezier':
    control_point = np.hstack((input_x, input_y, input_z))
    # ----------------initialization----------------
    input_vx = [0, 0]  # start and end vel for the trajectory
    input_ax = [0, 0]  # start and end acc for the trajectory
    planner = CubicBezier(time_sequence, control_point, input_vx, input_ax, 'vel')
    point_set_x, point_set_y, point_set_z = planner.two_point_to_four()
    # ------------------- Control loop --------------------
    state = con.receive()
    tcp = state.actual_TCP_pose
    t_current = 0
    t_start = state.timestamp
    print('t_start=', t_start)
    while state.timestamp - t_start < t_final:
        # t_init = state.timestamp
        state = con.receive()
        t_prev = t_current
        t_current = state.timestamp - t_start
        dt = t_current - t_prev
        if state.runtime_state > 1:
            #   ----------- minimum_jerk trajectory --------------
            if t_current <= t_final:
                print('t_c=', t_current)
                for i in range(len(time_sequence) - 1):
                    # time interval judgment
                    if time_sequence[i] <= t_current < time_sequence[i + 1]:
                        # return pos, vel, acc
                        res_x = planner.calc_cubic_Bezier(i, point_set_x[i], t_current)
                        res_y = planner.calc_cubic_Bezier(i, point_set_y[i], t_current)
                        res_z = planner.calc_cubic_Bezier(i, point_set_z[i], t_current)
                        position_ref = [res_x[0], res_y[0], res_z[0]]
                        velocity_ref = [res_x[1], res_y[1], res_z[1]]
                        acceleration_ref = [res_x[2], res_y[2], res_z[2]]
                        break
            # ------------------ impedance -----------------------
            state.actual_TCP_pose = state.actual_TCP_pose
            state.actual_TCP_speed = state.actual_TCP_speed
            current_force = state.actual_TCP_force
            t_current = state.timestamp - t_start
            pose = position_ref + start_pose[3:]  # Cubic Spline given to polsycope
            list_to_setp(setp, pose)
            con.send(setp)

            if plotter:
                time_plot.append(t_current)
                min_jerk_x.append(position_ref[0])
                min_jerk_y.append(position_ref[1])
                min_jerk_z.append(position_ref[2])

                min_jerk_vx.append(velocity_ref[0])
                min_jerk_vy.append(velocity_ref[1])
                min_jerk_vz.append(velocity_ref[2])

                min_jerk_ax.append(acceleration_ref[0])
                min_jerk_ay.append(acceleration_ref[1])
                min_jerk_az.append(acceleration_ref[2])

                px.append(state.actual_TCP_pose[0])
                py.append(state.actual_TCP_pose[1])
                pz.append(state.actual_TCP_pose[2])

                vx.append(state.actual_TCP_speed[0])
                vy.append(state.actual_TCP_speed[1])
                vz.append(state.actual_TCP_speed[2])

                if len(time_plot) <= 10:
                    # calc_vel_x.append(0)
                    # calc_vel_y.append(0)
                    # calc_vel_z.append(0)
                    calc_acc_x.append(0)
                    calc_acc_y.append(0)
                    calc_acc_z.append(0)
                else:
                    # dx = px[-1] - px[-2]
                    # dy = py[-1] - py[-2]
                    # dz = pz[-1] - pz[-2]
                    #
                    # calc_vel_x.append(dx/dt)
                    # calc_vel_y.append(dy/dt)
                    # calc_vel_z.append(dz/dt
                    dvx = vx[-1] - vx[-3]
                    dvy = vy[-1] - vy[-3]
                    dvz = vz[-1] - vz[-3]
                    calc_acc_x.append(dvx/(dt*2))
                    calc_acc_y.append(dvy/(dt*2))
                    calc_acc_z.append(dvz/(dt*2))

                joint_current_0.append(state.actual_current[0])
                joint_current_1.append(state.actual_current[1])
                joint_current_2.append(state.actual_current[2])
                joint_current_3.append(state.actual_current[3])
                joint_current_4.append(state.actual_current[4])
                joint_current_5.append(state.actual_current[5])

                # joint power: j_p
                j_p = np.array(state.actual_current) * voltage_const
                joint_power_0.append(j_p[0])
                joint_power_1.append(j_p[1])
                joint_power_2.append(j_p[2])
                joint_power_3.append(j_p[3])
                joint_power_4.append(j_p[4])
                joint_power_5.append(j_p[5])

                # accumulation of consumed energy
                # j_e: joint_energy
                j_e = abs(j_p) * dt
                joint_energy[0] += j_e[0]
                joint_energy[1] += j_e[1]
                joint_energy[2] += j_e[2]
                joint_energy[3] += j_e[3]
                joint_energy[4] += j_e[4]
                joint_energy[5] += j_e[5]

elif run_mode == 'LinearParabolic':
    num_rows = np.shape(input_x)[0]
    control_point = np.hstack((input_x, input_y, input_z))
    t_blend = 0.2*np.diff(time_sequence)[0]  # time for parabolic segments
    t_final = time_sequence[-1]
    planner = LinearParabolic(time_sequence, t_blend, control_point)
    M_lx, M_ly, M_lz = planner.co_line()
    M_px, M_py, M_pz = planner.co_parabolic()
    M_x = planner.co_linear_parabolic(M_lx, M_px)
    M_y = planner.co_linear_parabolic(M_ly, M_py)
    M_z = planner.co_linear_parabolic(M_lz, M_pz)
    time_interval = planner.time_interval
    # ------------------- Control loop --------------------
    state = con.receive()
    tcp = state.actual_TCP_pose
    t_current = 0
    t_start = state.timestamp
    print('t_start=', t_start)
    while state.timestamp - t_start < t_final:
        state = con.receive()
        t_prev = t_current
        t_current = state.timestamp - t_start
        dt = t_current - t_prev
        print(f"dt:{dt}")
        # read state from the robot
        if state.runtime_state > 1 and t_current <= t_final:
            #   ----------- minimum_jerk trajectory --------------
            print('t_c=', t_current)
            for i in range(2 * num_rows - 1):
                if time_interval[i] <= t_current < time_interval[i + 1]:
                    res_x = planner.calc_lp(M_x, i, t_current)
                    res_y = planner.calc_lp(M_y, i, t_current)
                    res_z = planner.calc_lp(M_z, i, t_current)
                    break
            position_ref = [res_x[0], res_y[0], res_z[0]]
            velocity_ref = [res_x[1], res_y[1], res_z[1]]
            acceleration_ref = [res_x[2], res_y[2], res_z[2]]
            # ------------------ impedance -----------------------
            state.actual_TCP_pose = state.actual_TCP_pose
            state.actual_TCP_speed = state.actual_TCP_speed
            current_force = state.actual_TCP_force
            t_current = state.timestamp - t_start
            pose = position_ref + rotation_const  # 6D vector of position given to polsycope
            list_to_setp(setp, pose)
            con.send(setp)

            if plotter:
                time_plot.append(t_current)
                min_jerk_x.append(position_ref[0])
                min_jerk_y.append(position_ref[1])
                min_jerk_z.append(position_ref[2])

                min_jerk_vx.append(velocity_ref[0])
                min_jerk_vy.append(velocity_ref[1])
                min_jerk_vz.append(velocity_ref[2])

                min_jerk_ax.append(acceleration_ref[0])
                min_jerk_ay.append(acceleration_ref[1])
                min_jerk_az.append(acceleration_ref[2])

                px.append(state.actual_TCP_pose[0])
                py.append(state.actual_TCP_pose[1])
                pz.append(state.actual_TCP_pose[2])

                vx.append(state.actual_TCP_speed[0])
                vy.append(state.actual_TCP_speed[1])
                vz.append(state.actual_TCP_speed[2])

                if len(time_plot) <= 10:
                    # calc_vel_x.append(0)
                    # calc_vel_y.append(0)
                    # calc_vel_z.append(0)
                    calc_acc_x.append(0)
                    calc_acc_y.append(0)
                    calc_acc_z.append(0)
                else:
                    # dx = px[-1] - px[-2]
                    # dy = py[-1] - py[-2]
                    # dz = pz[-1] - pz[-2]
                    #
                    # calc_vel_x.append(dx/dt)
                    # calc_vel_y.append(dy/dt)
                    # calc_vel_z.append(dz/dt
                    dvx = vx[-1] - vx[-3]
                    dvy = vy[-1] - vy[-3]
                    dvz = vz[-1] - vz[-3]
                    calc_acc_x.append(dvx/(dt*2))
                    calc_acc_y.append(dvy/(dt*2))
                    calc_acc_z.append(dvz/(dt*2))

                joint_current_0.append(state.actual_current[0])
                joint_current_1.append(state.actual_current[1])
                joint_current_2.append(state.actual_current[2])
                joint_current_3.append(state.actual_current[3])
                joint_current_4.append(state.actual_current[4])
                joint_current_5.append(state.actual_current[5])

                # joint power: j_p
                j_p = np.array(state.actual_current) * voltage_const
                joint_power_0.append(j_p[0])
                joint_power_1.append(j_p[1])
                joint_power_2.append(j_p[2])
                joint_power_3.append(j_p[3])
                joint_power_4.append(j_p[4])
                joint_power_5.append(j_p[5])

                # accumulation of consumed energy
                # j_e: joint_energy
                j_e = abs(j_p) * dt
                joint_energy[0] += j_e[0]
                joint_energy[1] += j_e[1]
                joint_energy[2] += j_e[2]
                joint_energy[3] += j_e[3]
                joint_energy[4] += j_e[4]
                joint_energy[5] += j_e[5]

elif run_mode == 'Spline535':
    control_point = np.hstack((input_x, input_y, input_z))
    planner = Spline535(time_sequence, control_point)
    # -----------------------user defined value---------------------------
    # here users can define  v,a,j at start and end point and knot velocity
    if planner.num_rows == 5:
        setter = 1
        '''user can define value under "if setter" '''
        if setter:
            # -------------------- x axis---------------------------
            # start pos, vel, acc, jerk
            qx_0, vx_0, ax_0, jx_0 = list(input_x[0]), 0, 0, 0
            # way point pos, vel
            qx_1, vx_1, qx_2, vx_2, qx_3, vx_3 = list(input_x[1]), 0, list(input_x[2]), 0, list(input_x[3]), 0
            # end pos, vel, acc, jerk
            qx_4, vx_4, ax_4, jx_4 = list(input_x[4]), 0, 0, 0
            # -------------------- y axis---------------------------
            qy_0, vy_0, ay_0, jy_0 = list(input_y[0]), 0, 0, 0
            # way point pos, vel
            qy_1, vy_1, qy_2, vy_2, qy_3, vy_3 = list(input_y[1]), 0, list(input_y[2]), 0, list(input_y[3]), 0
            # end pos, vel, acc, jerk
            qy_4, vy_4, ay_4, jy_4 = list(input_y[4]), 0, 0, 0
            # -------------------- z axis---------------------------
            qz_0, vz_0, az_0, jz_0 = list(input_z[0]), 0, 0, 0
            # waz point pos, vel
            qz_1, vz_1, qz_2, vz_2, qz_3, vz_3 = list(input_z[1]), 0, list(input_z[2]), 0, list(input_z[3]), 0
            # end pos, vel, acc, jerk
            qz_4, vz_4, az_4, jz_4 = list(input_z[4]), 0, 0, 0
            # right side value of coefficient equations:  a0 + a1t + ...+ a5t**5 = r_value
            # -------------------- x axis---------------------------
            r_value_x = [qx_0, vx_0, ax_0, jx_0] + [0] * 8 + [qx_1, vx_1] + [0] * 6 + \
                        [0] * 8 + [qx_2, vx_2] + [0] * 6 + \
                        [0] * 8 + [qx_3, vx_3] + [0] * 6 + \
                        [0] * 8 + [qx_4, vx_4, ax_4, jx_4]
            r_value_x = [item[0] if isinstance(item, list) else item for item in r_value_x]
            # -------------------- y axis---------------------------
            r_value_y = [qy_0, vy_0, ay_0, jy_0] + [0] * 8 + [qy_1, vy_1] + [0] * 6 + \
                        [0] * 8 + [qy_2, vy_2] + [0] * 6 + \
                        [0] * 8 + [qy_3, vy_3] + [0] * 6 + \
                        [0] * 8 + [qy_4, vy_4, ay_4, jy_4]
            r_value_y = [item[0] if isinstance(item, list) else item for item in r_value_y]
            # -------------------- z axis---------------------------
            r_value_z = [qz_0, vz_0, az_0, jz_0] + [0] * 8 + [qz_1, vz_1] + [0] * 6 + \
                        [0] * 8 + [qz_2, vz_2] + [0] * 6 + \
                        [0] * 8 + [qz_3, vz_3] + [0] * 6 + \
                        [0] * 8 + [qz_4, vz_4, az_4, jz_4]
            r_value_z = [item[0] if isinstance(item, list) else item for item in r_value_z]
        # for way points=3, choose co_calculate_wp3()
        M_x = planner.co_calculate_wp3(r_value_x)
        M_y = planner.co_calculate_wp3(r_value_y)
        M_z = planner.co_calculate_wp3(r_value_z)
        time_interval = planner.time_interval
    elif planner.num_rows == 4:
        setter = 1
        '''user can define value under "if setter" '''
        if setter:
            # -----------------------user defined value---------------------------
            # ----------------------- x axis ---------------------------
            '''define value at start point: pos(from input_xyz), vel, acc ,jerk'''
            qx_0, vx_0, ax_0, jx_0 = list(input_x[0]), 0, 0, 0
            '''define value at way point: pos(from input_xyz), vel'''
            qx_1, vx_1, qx_2, vx_2 = list(input_x[1]), 0, list(input_x[2]), 0                         # SET the Velocities of Waypoints here
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
        # for way points=2, choose co_calculate_wp2()
        M_x = planner.co_calculate_wp2(r_value_x)
        M_y = planner.co_calculate_wp2(r_value_y)
        M_z = planner.co_calculate_wp2(r_value_z)
        time_interval = planner.time_interval
    # -----------------------user defined value end---------------------------
    # ------------------- Control loop --------------------
    state = con.receive()
    t_start = state.timestamp
    tcp = state.actual_TCP_pose
    t_current = 0
    print('t_start=', t_start)

    while state.timestamp - t_start < t_final:
        state = con.receive()
        t_prev = t_current
        t_current = state.timestamp - t_start
        dt = t_current - t_prev
        print(f"dt:{dt}")
        # read state from the robot
        if state.runtime_state > 1 and t_current <= t_final:
            #   ----------- minimum_jerk trajectory --------------
            print('t_c=', t_current)
            for i in range(3 * planner.num_rows - 3):
                if time_interval[i] <= t_current < time_interval[i + 1]:
                    res_x = planner.calc_535(M_x[i], t_current)
                    res_y = planner.calc_535(M_y[i], t_current)
                    res_z = planner.calc_535(M_z[i], t_current)
                    break
            position_ref = [res_x[0], res_y[0], res_z[0]]
            velocity_ref = [res_x[1], res_y[1], res_z[1]]
            acceleration_ref = [res_x[2], res_y[2], res_z[2]]
            jerk_ref = [res_x[3], res_y[3], res_z[3]]
            # ------------------ impedance -----------------------
            state.actual_TCP_pose = state.actual_TCP_pose
            state.actual_TCP_speed = state.actual_TCP_speed
            current_force = state.actual_TCP_force
            t_current = state.timestamp - t_start

            pose = position_ref + rotation_const  # 6D vector of position given to polsycope
            list_to_setp(setp, pose)
            con.send(setp)

            if plotter:
                time_plot.append(t_current)
                min_jerk_x.append(position_ref[0])
                min_jerk_y.append(position_ref[1])
                min_jerk_z.append(position_ref[2])

                min_jerk_vx.append(velocity_ref[0])
                min_jerk_vy.append(velocity_ref[1])
                min_jerk_vz.append(velocity_ref[2])

                min_jerk_ax.append(acceleration_ref[0])
                min_jerk_ay.append(acceleration_ref[1])
                min_jerk_az.append(acceleration_ref[2])

                px.append(state.actual_TCP_pose[0])
                py.append(state.actual_TCP_pose[1])
                pz.append(state.actual_TCP_pose[2])

                vx.append(state.actual_TCP_speed[0])
                vy.append(state.actual_TCP_speed[1])
                vz.append(state.actual_TCP_speed[2])

                if len(time_plot) <= 10:
                    # calc_vel_x.append(0)
                    # calc_vel_y.append(0)
                    # calc_vel_z.append(0)
                    calc_acc_x.append(0)
                    calc_acc_y.append(0)
                    calc_acc_z.append(0)
                else:
                    # dx = px[-1] - px[-2]
                    # dy = py[-1] - py[-2]
                    # dz = pz[-1] - pz[-2]
                    #
                    # calc_vel_x.append(dx/dt)
                    # calc_vel_y.append(dy/dt)
                    # calc_vel_z.append(dz/dt
                    dvx = vx[-1] - vx[-3]
                    dvy = vy[-1] - vy[-3]
                    dvz = vz[-1] - vz[-3]
                    calc_acc_x.append(dvx/(dt*2))
                    calc_acc_y.append(dvy/(dt*2))
                    calc_acc_z.append(dvz/(dt*2))

                joint_current_0.append(state.actual_current[0])
                joint_current_1.append(state.actual_current[1])
                joint_current_2.append(state.actual_current[2])
                joint_current_3.append(state.actual_current[3])
                joint_current_4.append(state.actual_current[4])
                joint_current_5.append(state.actual_current[5])

                # joint power: j_p
                j_p = np.array(state.actual_current) * voltage_const
                joint_power_0.append(j_p[0])
                joint_power_1.append(j_p[1])
                joint_power_2.append(j_p[2])
                joint_power_3.append(j_p[3])
                joint_power_4.append(j_p[4])
                joint_power_5.append(j_p[5])

                # accumulation of consumed energy
                # j_e: joint_energy
                j_e = abs(j_p) * dt
                joint_energy[0] += j_e[0]
                joint_energy[1] += j_e[1]
                joint_energy[2] += j_e[2]
                joint_energy[3] += j_e[3]
                joint_energy[4] += j_e[4]
                joint_energy[5] += j_e[5]

elif run_mode == 'QuinticBezier':
    # ----------------initialization----------------
    control_point = np.hstack((input_x, input_y, input_z))
    # vel and acc are all the same for x-y-z axis
    input_vx = [[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]]
    input_ax = [[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]]
    t_final = time_sequence[-1]
    planner = QuinticBezier(time_sequence, control_point, input_vx, input_ax)
    point_set_x, point_set_y, point_set_z = planner.two_point_to_six()
    # ------------------- Control loop --------------------
    state = con.receive()
    t_start = state.timestamp
    tcp = state.actual_TCP_pose
    t_current = 0
    print('t_start=', t_start)

    while state.timestamp - t_start < t_final:
        state = con.receive()
        t_prev = t_current
        t_current = state.timestamp - t_start
        dt = t_current - t_prev
        print(f"dt:{dt}")
        if state.runtime_state > 1 and t_current <= t_final:
            # ------------- minimum_jerk trajectory --------------
            for i in range(planner.num_rows - 1):
                # time interval judgment
                if time_sequence[i] <= t_current < time_sequence[i + 1]:
                    # return pos, vel, acc
                    res_x = planner.calc_quintic_Bezier(i, point_set_x[i], t_current)
                    res_y = planner.calc_quintic_Bezier(i, point_set_y[i], t_current)
                    res_z = planner.calc_quintic_Bezier(i, point_set_z[i], t_current)
                    position_ref = [res_x[0], res_y[0], res_z[0]]
                    velocity_ref = [res_x[1], res_y[1], res_z[1]]
                    acceleration_ref = [res_x[2], res_y[2], res_z[2]]
                    # rotation_ref = [rotation_x[i+1], rotation_y[i+1], rotation_z[i+1]]
                    break
            # ------------------ impedance -----------------------
            # state.actual_TCP_pose
            # state.actual_TCP_speed
            t_current = state.timestamp - t_start
            pose = position_ref + rotation_const  # 6D vector of position given to polsycope
            list_to_setp(setp, pose)
            con.send(setp)

            if plotter:
                time_plot.append(t_current)
                min_jerk_x.append(position_ref[0])
                min_jerk_y.append(position_ref[1])
                min_jerk_z.append(position_ref[2])

                min_jerk_vx.append(velocity_ref[0])
                min_jerk_vy.append(velocity_ref[1])
                min_jerk_vz.append(velocity_ref[2])

                min_jerk_ax.append(acceleration_ref[0])
                min_jerk_ay.append(acceleration_ref[1])
                min_jerk_az.append(acceleration_ref[2])

                px.append(state.actual_TCP_pose[0])
                py.append(state.actual_TCP_pose[1])
                pz.append(state.actual_TCP_pose[2])

                vx.append(state.actual_TCP_speed[0])
                vy.append(state.actual_TCP_speed[1])
                vz.append(state.actual_TCP_speed[2])

                if len(time_plot) <= 10:
                    calc_vel_x.append(0)
                    calc_vel_y.append(0)
                    calc_vel_z.append(0)
                    calc_acc_x.append(0)
                    calc_acc_y.append(0)
                    calc_acc_z.append(0)
                else:
                    dx = px[-1] - px[-2]
                    dy = py[-1] - py[-2]
                    dz = pz[-1] - pz[-2]

                    calc_vel_x.append(dx/dt)
                    calc_vel_y.append(dy/dt)
                    calc_vel_z.append(dz/dt)
                    dvx = vx[-1] - vx[-3]
                    dvy = vy[-1] - vy[-3]
                    dvz = vz[-1] - vz[-3]
                    calc_acc_x.append(dvx / (dt * 2))
                    calc_acc_y.append(dvy / (dt * 2))
                    calc_acc_z.append(dvz / (dt * 2))

                joint_current_0.append(state.actual_current[0])
                joint_current_1.append(state.actual_current[1])
                joint_current_2.append(state.actual_current[2])
                joint_current_3.append(state.actual_current[3])
                joint_current_4.append(state.actual_current[4])
                joint_current_5.append(state.actual_current[5])

                # joint power: j_p
                j_p = np.array(state.actual_current) * voltage_const
                joint_power_0.append(j_p[0])
                joint_power_1.append(j_p[1])
                joint_power_2.append(j_p[2])
                joint_power_3.append(j_p[3])
                joint_power_4.append(j_p[4])
                joint_power_5.append(j_p[5])

                # accumulation of consumed energy
                # j_e: joint_energy
                j_e = abs(j_p) * dt
                joint_energy[0] += j_e[0]
                joint_energy[1] += j_e[1]
                joint_energy[2] += j_e[2]
                joint_energy[3] += j_e[3]
                joint_energy[4] += j_e[4]
                joint_energy[5] += j_e[5]

print(f"It took {state.timestamp}s to execute the servoJ")
print(f"time needed for min_jerk {t_final}\n")
state = con.receive()
print('--------------------')
print('state.actual_TCP_pose\n', state.actual_TCP_pose)
print('consumed energy for each joint, unit: [J]\n', joint_energy)

# ====================mode 2===================
watchdog.input_int_register_0 = 2
con.send(watchdog)
con.send_pause()
con.disconnect()

'''save npy'''
save_npy = 0
if save_npy:
    # Step 1: Define the folder path
    folder_path = r'C:\Users\Kerouac\Desktop\project_UR3e\Servoj_RTDE_UR3-main\task_space_' + run_mode  # You can change this to any folder name or path you prefer

    # Step 2: Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # print(f"Saving to file: {run_mode+' task_position_xyz_robot.npy'}")
    np.save(os.path.join(folder_path, run_mode + ' position_xyz_robot.npy'), [px, py, pz])

    # print(f"Saving to file: {run_mode+' task_velocity_xyz_robot.npy'}")
    np.save(os.path.join(folder_path, run_mode+' velocity_xyz_robot'), [vx, vy, vz])

    # print(f"Saving to file: {run_mode+' task_acceleration_xyz_calc.npy'}")
    np.save(os.path.join(folder_path, run_mode+' acceleration_xyz_calc'), [calc_acc_x, calc_acc_y, calc_acc_z])

    # print(f"Saving to file: {run_mode+' task_acceleration_xyz_planned.npy'}")
    np.save(os.path.join(folder_path, run_mode+' acceleration_xyz_planned'), [min_jerk_ax, min_jerk_ay, min_jerk_az])

    # print(f"Saving to file: {run_mode+' task_space_measured_current.npy'}")
    np.save(os.path.join(folder_path, run_mode+' measured_current'), [joint_current_0, joint_current_1, joint_current_2, joint_current_3, joint_current_4, joint_current_5])

    # print(f"Saving to file: {run_mode + ' joint_space_power.npy'}")
    np.save(os.path.join(folder_path, run_mode + ' power'), [joint_power_0, joint_power_1, joint_power_2, joint_power_3, joint_power_4, joint_power_5])

    # print(f"Saving to file: {run_mode+' task_space_consumed_energy.npy'}")
    np.save(os.path.join(folder_path, run_mode+' consumed energy'), joint_energy)

    print(f'{run_mode}:npy_data saved successfully')

'''plot'''
plot_auto_save = 0
if plotter:
    # ----------- position -------------
    fig = plt.figure(0)
    plt.title(run_mode + '_xyz')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(input_x[0], input_y[0], input_z[0], color='red', label='start point')
    ax.scatter(input_x[-1], input_y[-1], input_z[-1], color='blue', label='desired point')
    ax.scatter(input_x[1:-1], input_y[1:-1], input_z[1:-1], color='green', label='waypoint')
    ax.plot3D(min_jerk_x, min_jerk_y, min_jerk_z, color='orange', alpha=0.5, linewidth=3, label='planned trajectory')
    ax.plot3D(px, py, pz, color='green', label='actual trajectory')
    ax.set_xlabel('X position [m]')
    ax.set_ylabel('Y position [m]')
    ax.set_zlabel('Z position [m]')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.figure(1)
    plt.title(run_mode + ' position_x')
    plt.plot(time_plot, px, label="x_robot")
    plt.plot(time_plot, min_jerk_x, label="x_planned")
    plt.plot(time_sequence, input_x, 'o', label='control point x coordinate')
    plt.legend()
    plt.grid()
    plt.ylabel('Position in x-y-z[m]')
    plt.xlabel('Time [sec]')


    plt.figure(2)
    plt.title(run_mode + ' position_y')
    plt.plot(time_plot, py, label="y_robot")
    plt.plot(time_plot, min_jerk_y, label="y_planned")
    plt.plot(time_sequence, input_y, 'o',  label='control point y coordinate')
    plt.legend()
    plt.grid()
    plt.ylabel('Position in y[m]')
    plt.xlabel('Time [sec]')

    plt.figure(3)
    plt.title(run_mode + ' position_z')
    plt.plot(time_plot, pz, label="z_robot")
    plt.plot(time_plot, min_jerk_z, label="z_planned")
    plt.plot(time_sequence, input_z, 'o', label='control point z coordinate')
    plt.legend()
    plt.grid()
    plt.ylabel('Position in z[m]')
    plt.xlabel('Time [sec]')

    # ----------- velocity -------------
    plt.figure(4)
    plt.title(run_mode + ' velocity_x')
    plt.plot(time_plot, calc_vel_x, label="vx_calc")
    plt.plot(time_plot, vx, label="vx_robot")
    plt.plot(time_plot, min_jerk_vx, label="vx_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time [sec]')

    plt.figure(5)
    plt.title(run_mode + ' velocity_y')
    plt.plot(time_plot, calc_vel_y, label="vy_calc")
    plt.plot(time_plot, vy, label="vy_robot")
    plt.plot(time_plot, min_jerk_vy, label="vy_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time [sec]')

    plt.figure(6)
    plt.title(run_mode + ' velocity_z')
    plt.plot(time_plot, calc_vel_z, label="vz_calc")
    plt.plot(time_plot, vz, label="vz_robot")
    plt.plot(time_plot, min_jerk_vz, label="vz_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time [sec]')
    # ----------- Acceleration -------------
    plt.figure(7)
    plt.title(run_mode + ' acceleration')
    plt.plot(time_plot, calc_acc_x, label="ax_measured")
    plt.plot(time_plot, min_jerk_ax, label="ax_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Acceleration [m/s^2]')
    plt.xlabel('Time [sec]')

    plt.figure(8)
    plt.title(run_mode + ' acceleration')
    plt.plot(time_plot, calc_acc_y, label="ay_measured")
    plt.plot(time_plot, min_jerk_ay, label="ay_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Acceleration [m/s^2]')
    plt.xlabel('Time [sec]')

    plt.figure(9)
    plt.title(run_mode + ' acceleration')
    plt.plot(time_plot, calc_acc_z, label="az_measured")
    plt.plot(time_plot, min_jerk_az, label="az_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Acceleration [m/s^2]')
    plt.xlabel('Time [sec]')

    # ----------- Current -------------
    plt.figure(10)
    plt.title(run_mode + ' joint_current')
    plt.plot(time_plot, joint_current_0, label="actual_current_0")
    plt.plot(time_plot, joint_current_1, label="actual_current_1")
    plt.plot(time_plot, joint_current_2, label="actual_current_2")
    plt.plot(time_plot, joint_current_3, label="actual_current_3")
    plt.plot(time_plot, joint_current_4, label="actual_current_4")
    plt.plot(time_plot, joint_current_5, label="actual_current_5")
    plt.legend()
    plt.grid()
    plt.ylabel('Current [A]')
    plt.xlabel('Time [sec]')
    # ----------- Power -------------
    plt.figure(11)
    plt.title(run_mode + ' joint_power')
    plt.plot(time_plot, joint_power_0, label="actual_power_0")
    plt.plot(time_plot, joint_power_1, label="actual_power_1")
    plt.plot(time_plot, joint_power_2, label="actual_power_2")
    plt.plot(time_plot, joint_power_3, label="actual_power_3")
    plt.plot(time_plot, joint_power_4, label="actual_power_4")
    plt.plot(time_plot, joint_power_5, label="actual_power_5")
    plt.legend()
    plt.grid()
    plt.ylabel('Power [w]')
    plt.xlabel('Time [sec]')
    # ----------- Energy -------------
    plt.figure(12)
    plt.title(run_mode + ' joint_energy')
    categories = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
    plt.bar(categories, joint_energy, label='consumed energy')
    plt.legend()
    plt.grid()
    plt.xlabel('joints')
    plt.ylabel('consumed energy')
    # ----------- auto save -------------
    if plot_auto_save:
        output_dir = r'C:\Users\Kerouac\Desktop\plot_task_space_' + run_mode
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # 如果目录不存在则创建

        # 获取当前所有生成的图形编号
        figures = plt.get_fignums()  # 返回所有图形编号的列表
        # 遍历所有生成的图形并保存
        for i, fig_num in enumerate(figures):
            fig = plt.figure(fig_num)  # 获取当前编号的图形
            file_path = os.path.join(output_dir, f'_task_space_{run_mode}_plot_{i}.svg')  # 生成文件名
            fig.savefig(file_path, format='svg')  # 保存图像
            # print(f"Plot {i + 1} saved to {file_path}")
        print("All plots have been saved.")

    plt.show()


