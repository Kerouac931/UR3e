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
from joint_class_LinearParabolic_joint import LinearParabolic
from joint_class_CubicBezier_joint import CubicBezier
from joint_class_CubicSpline_joint import CubicSpline
from joint_class_Spline535_joint import Spline535
from joint_class_QuinticBezier_joint import QuinticBezier
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
# start_pose = [0.00415, -0.261, 0.441, 0.269, -2.85, 0.69]
# desired_pose = [0.27788, -0.30054, 0.35810, 0.094, -3.04, 0.706]
# -----------------------data set 1---------------------------
# control point at TCP postion , unit:m
# input_x = np.array([5.05, 42.98, 301.28, 277.88]).reshape(4, 1) * 0.001
# input_y = np.array([-260.95, -320.32, -276.68, -300.54]).reshape(4, 1) * 0.001
# input_z = np.array([440.64, 313.66, 317.05, 358.10]).reshape(4, 1) * 0.001
input_x = np.array([5.05, 42.98, 150.15, 277.88]).reshape(4, 1) * 0.001
input_y = np.array([-260.95, -250.25, -276.68, -300.54]).reshape(4, 1) * 0.001
input_z = np.array([440.64, 380.38, 317.05, 358.10]).reshape(4, 1) * 0.001
rotation_x = [0.094, 0.094, 0.094, 0.094]
rotation_y = [-3.035, -3.04, -3.04, -3.04]
rotation_z = [0.706, 0.706, 0.706, 0.706]

# this is joint angle in radian
# 0.01745=2*pi/360 from degree to rad
input_q0 = np.array([-52.08, -40.55, -29.56, -22.91]).reshape(4, 1)*0.01745
input_q1 = np.array([-69.89, -62.23, -78.51, -105.22]).reshape(4, 1)*0.01745
input_q2 = np.array([-78.37, -98.58, -104.31, -64.42]).reshape(4, 1)*0.01745
input_q3 = np.array([-100.74, -91.70, -73.82, -89.82]).reshape(4, 1)*0.01745
input_q4 = np.array([74.04, 70.23, 67.29, 65.92]).reshape(4, 1)*0.01745
input_q5 = np.array([31.40, 42.84, 54.21, 61.29]).reshape(4, 1)*0.01745

time_sequence = np.linspace(0, 9, len(input_q0))
t_final = time_sequence[-1]  # final time of the trajectory

voltage_const = 48
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
con.send(watchdog)  # sending mode == 1
dt = 1 / FREQUENCY  # 500 Hz    # frequency
plotter = True
# initialization of plot
if plotter:
    time_plot = []
    min_jerk_q0, min_jerk_q1, min_jerk_q2 = [], [], []
    min_jerk_q3, min_jerk_q4, min_jerk_q5 = [], [], []
    min_jerk_d_q0, min_jerk_d_q1, min_jerk_d_q2 = [], [], []
    min_jerk_d_q3, min_jerk_d_q4, min_jerk_d_q5 = [], [], []
    min_jerk_d2_q0, min_jerk_d2_q1, min_jerk_d2_q2 = [], [], []
    min_jerk_d2_q3, min_jerk_d2_q4, min_jerk_d2_q5 = [], [], []
    q0, q1, q2, q3, q4, q5 = [], [], [], [], [], []
    d_q0, d_q1, d_q2, d_q3, d_q4, d_q5 = [], [], [], [], [], []
    d2_q0, d2_q1, d2_q2, d2_q3, d2_q4, d2_q5 = [], [], [], [], [], []
    planned_px, planned_py, planned_pz = [], [], []
    planned_vx, planned_vy, planned_vz = [], [], []
    planned_ax, planned_ay, planned_az = [], [], []
    px, py, pz, vx, vy, vz, force_x, force_y, force_z = [], [], [], [], [], [], [], [], []
    joint_current_0, joint_current_1, joint_current_2 = [], [], []
    joint_current_3, joint_current_4, joint_current_5 = [], [], []
    joint_power_0, joint_power_1, joint_power_2 = [], [], []
    joint_power_3, joint_power_4, joint_power_5 = [], [], []
    joint_energy = np.zeros(6)
    calc_acc_0, calc_acc_1, calc_acc_2, calc_acc_3, calc_acc_4, calc_acc_5 = [], [], [], [], [], []
    time_range = []

# ---------------- choose algorithm ----------------
#                 0             1                 2               3               4
mode_list = ['CubicSpline', 'CubicBezier', 'LinearParabolic', 'Spline535', 'QuinticBezier']
# run_mode[]: it matches to mode_list, choose different trajectory algorithm
run_mode = mode_list[4]

if run_mode == 'CubicSpline':
    # ------------------ Control loop initialization -------------------------
    control_point = np.hstack((input_q0, input_q1, input_q2, input_q3, input_q4, input_q5))
    # v0 and vf is where users define the start and end velocity
    v0, vf = 0, 0
    a0, af = 0, 0
    planner = CubicSpline(time_sequence, control_point, [v0, vf], [a0, af], 'vel')
    # res_cubic_spline = cubic_spline(time_sequence, control_point)
    # calc the coefficients of every segment of curve for xyz axis
    co_q0 = planner.calculate_coefficient(input_q0)
    co_q1 = planner.calculate_coefficient(input_q1)
    co_q2 = planner.calculate_coefficient(input_q2)
    co_q3 = planner.calculate_coefficient(input_q3)
    co_q4 = planner.calculate_coefficient(input_q4)
    co_q5 = planner.calculate_coefficient(input_q5)
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
        if state.runtime_state > 1:
            #   ----------- minimum_jerk trajectory --------------
            if t_current <= t_final:
                print('t_c=', t_current)
                # [position_ref, lin_vel_ref, acceleration_ref] = planner.trajectory_planning(t_current)
                res_q0 = planner.calculate_with_t_current(co_q0, t_current)
                res_q1 = planner.calculate_with_t_current(co_q1, t_current)
                res_q2 = planner.calculate_with_t_current(co_q2, t_current)
                res_q3 = planner.calculate_with_t_current(co_q3, t_current)
                res_q4 = planner.calculate_with_t_current(co_q4, t_current)
                res_q5 = planner.calculate_with_t_current(co_q5, t_current)
                position_ref = [res_q0[0], res_q1[0], res_q2[0], res_q3[0], res_q4[0], res_q5[0]]
                velocity_ref = [res_q0[1], res_q1[1], res_q2[1], res_q3[1], res_q4[1], res_q5[1]]
                acceleration_ref = [res_q0[2], res_q1[2], res_q2[2], res_q3[2], res_q4[2], res_q5[2]]
            # ------------------ impedance -----------------------
            # current_joint_pose = state.actual_q
            # current_joint_speed = state.actual_qd
            # current_joint_current = state.actual_current  # joint current
            # current_TCP_pose = state.actual_TCP_pose
            # current_TCP_speed = state.actual_TCP_speed
            list_to_setp(setp, position_ref)
            con.send(setp)

            if plotter:
                time_plot.append(state.timestamp - t_start)
                min_jerk_q0.append(position_ref[0] * 57.3248)
                min_jerk_q1.append(position_ref[1] * 57.3248)
                min_jerk_q2.append(position_ref[2] * 57.3248)
                min_jerk_q3.append(position_ref[3] * 57.3248)
                min_jerk_q4.append(position_ref[4] * 57.3248)
                min_jerk_q5.append(position_ref[5] * 57.3248)

                min_jerk_d_q0.append(velocity_ref[0] * 57.3248)
                min_jerk_d_q1.append(velocity_ref[1] * 57.3248)
                min_jerk_d_q2.append(velocity_ref[2] * 57.3248)
                min_jerk_d_q3.append(velocity_ref[3] * 57.3248)
                min_jerk_d_q4.append(velocity_ref[4] * 57.3248)
                min_jerk_d_q5.append(velocity_ref[5] * 57.3248)

                min_jerk_d2_q0.append(acceleration_ref[0] * 57.3248)
                min_jerk_d2_q1.append(acceleration_ref[1] * 57.3248)
                min_jerk_d2_q2.append(acceleration_ref[2] * 57.3248)
                min_jerk_d2_q3.append(acceleration_ref[3] * 57.3248)
                min_jerk_d2_q4.append(acceleration_ref[4] * 57.3248)
                min_jerk_d2_q5.append(acceleration_ref[5] * 57.3248)

                q0.append(state.actual_q[0] * 57.3248)
                q1.append(state.actual_q[1] * 57.3248)
                q2.append(state.actual_q[2] * 57.3248)
                q3.append(state.actual_q[3] * 57.3248)
                q4.append(state.actual_q[4] * 57.3248)
                q5.append(state.actual_q[5] * 57.3248)

                d_q0.append(state.actual_qd[0] * 57.3248)
                d_q1.append(state.actual_qd[1] * 57.3248)
                d_q2.append(state.actual_qd[2] * 57.3248)
                d_q3.append(state.actual_qd[3] * 57.3248)
                d_q4.append(state.actual_qd[4] * 57.3248)
                d_q5.append(state.actual_qd[5] * 57.3248)

                if len(time_plot) <= 10:
                    calc_acc_0.append(0)
                    calc_acc_1.append(0)
                    calc_acc_2.append(0)
                    calc_acc_3.append(0)
                    calc_acc_4.append(0)
                    calc_acc_5.append(0)
                else:
                    dv0 = d_q0[-1] - d_q0[-3]
                    dv1 = d_q1[-1] - d_q1[-3]
                    dv2 = d_q2[-1] - d_q2[-3]
                    dv3 = d_q3[-1] - d_q3[-3]
                    dv4 = d_q4[-1] - d_q4[-3]
                    dv5 = d_q5[-1] - d_q5[-3]
                    calc_acc_0.append(dv0 / (dt*2))
                    calc_acc_1.append(dv1 / (dt*2))
                    calc_acc_2.append(dv2 / (dt*2))
                    calc_acc_3.append(dv3 / (dt*2))
                    calc_acc_4.append(dv4 / (dt*2))
                    calc_acc_5.append(dv5 / (dt*2))

                px.append(state.actual_TCP_pose[0])
                py.append(state.actual_TCP_pose[1])
                pz.append(state.actual_TCP_pose[2])

                vx.append(state.actual_TCP_speed[0])
                vy.append(state.actual_TCP_speed[1])
                vz.append(state.actual_TCP_speed[2])

                joint_current_0.append(state.actual_current[0])
                joint_current_1.append(state.actual_current[1])
                joint_current_2.append(state.actual_current[2])
                joint_current_3.append(state.actual_current[3])
                joint_current_4.append(state.actual_current[4])
                joint_current_5.append(state.actual_current[5])

                #joint power: j_p
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
    control_point = np.hstack((input_q0, input_q1, input_q2, input_q3, input_q4, input_q5))
    # ----------------initialization----------------
    input_d_q = [0, 0]  # start and end vel for the trajectory
    input_d2_q = [0, 0]  # start and end acc for the trajectory
    planner = CubicBezier(time_sequence, control_point, input_d_q, input_d2_q, 'vel')
    point_set_q0, point_set_q1, point_set_q2, point_set_q3, point_set_q4, point_set_q5 = planner.two_point_to_four()
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
                print('dt=', dt)
                for i in range(len(time_sequence) - 1):
                    # time interval judgment
                    if time_sequence[i] <= t_current < time_sequence[i + 1]:
                        # return pos, vel, acc
                        res_q0 = planner.calc_cubic_Bezier(i, point_set_q0[i], t_current)
                        res_q1 = planner.calc_cubic_Bezier(i, point_set_q1[i], t_current)
                        res_q2 = planner.calc_cubic_Bezier(i, point_set_q2[i], t_current)
                        res_q3 = planner.calc_cubic_Bezier(i, point_set_q3[i], t_current)
                        res_q4 = planner.calc_cubic_Bezier(i, point_set_q4[i], t_current)
                        res_q5 = planner.calc_cubic_Bezier(i, point_set_q5[i], t_current)
                        position_ref = [res_q0[0], res_q1[0], res_q2[0], res_q3[0], res_q4[0], res_q5[0]]
                        velocity_ref = [res_q0[1], res_q1[1], res_q2[1], res_q3[1], res_q4[1], res_q5[1]]
                        acceleration_ref = [res_q0[2], res_q1[2], res_q2[2], res_q3[2], res_q4[2], res_q5[2]]
                        break
            # ------------------ impedance -----------------------
            # current_joint_pose = state.actual_q
            # current_joint_speed = state.actual_qd
            # current_joint_current = state.actual_current  # joint current
            # current_TCP_pose = state.actual_TCP_pose
            # current_TCP_speed = state.actual_TCP_speed

            list_to_setp(setp, position_ref)
            con.send(setp)

            if plotter:
                time_plot.append(state.timestamp - t_start)
                min_jerk_q0.append(position_ref[0] * 57.3248)
                min_jerk_q1.append(position_ref[1] * 57.3248)
                min_jerk_q2.append(position_ref[2] * 57.3248)
                min_jerk_q3.append(position_ref[3] * 57.3248)
                min_jerk_q4.append(position_ref[4] * 57.3248)
                min_jerk_q5.append(position_ref[5] * 57.3248)

                min_jerk_d_q0.append(velocity_ref[0] * 57.3248)
                min_jerk_d_q1.append(velocity_ref[1] * 57.3248)
                min_jerk_d_q2.append(velocity_ref[2] * 57.3248)
                min_jerk_d_q3.append(velocity_ref[3] * 57.3248)
                min_jerk_d_q4.append(velocity_ref[4] * 57.3248)
                min_jerk_d_q5.append(velocity_ref[5] * 57.3248)

                min_jerk_d2_q0.append(acceleration_ref[0] * 57.3248)
                min_jerk_d2_q1.append(acceleration_ref[1] * 57.3248)
                min_jerk_d2_q2.append(acceleration_ref[2] * 57.3248)
                min_jerk_d2_q3.append(acceleration_ref[3] * 57.3248)
                min_jerk_d2_q4.append(acceleration_ref[4] * 57.3248)
                min_jerk_d2_q5.append(acceleration_ref[5] * 57.3248)

                q0.append(state.actual_q[0] * 57.3248)
                q1.append(state.actual_q[1] * 57.3248)
                q2.append(state.actual_q[2] * 57.3248)
                q3.append(state.actual_q[3] * 57.3248)
                q4.append(state.actual_q[4] * 57.3248)
                q5.append(state.actual_q[5] * 57.3248)

                d_q0.append(state.actual_qd[0] * 57.3248)
                d_q1.append(state.actual_qd[1] * 57.3248)
                d_q2.append(state.actual_qd[2] * 57.3248)
                d_q3.append(state.actual_qd[3] * 57.3248)
                d_q4.append(state.actual_qd[4] * 57.3248)
                d_q5.append(state.actual_qd[5] * 57.3248)

                if len(time_plot) <= 10:
                    calc_acc_0.append(0)
                    calc_acc_1.append(0)
                    calc_acc_2.append(0)
                    calc_acc_3.append(0)
                    calc_acc_4.append(0)
                    calc_acc_5.append(0)
                else:
                    dv0 = d_q0[-1] - d_q0[-3]
                    dv1 = d_q1[-1] - d_q1[-3]
                    dv2 = d_q2[-1] - d_q2[-3]
                    dv3 = d_q3[-1] - d_q3[-3]
                    dv4 = d_q4[-1] - d_q4[-3]
                    dv5 = d_q5[-1] - d_q5[-3]
                    calc_acc_0.append(dv0 / (dt * 2))
                    calc_acc_1.append(dv1 / (dt * 2))
                    calc_acc_2.append(dv2 / (dt * 2))
                    calc_acc_3.append(dv3 / (dt * 2))
                    calc_acc_4.append(dv4 / (dt * 2))
                    calc_acc_5.append(dv5 / (dt * 2))

                px.append(state.actual_TCP_pose[0])
                py.append(state.actual_TCP_pose[1])
                pz.append(state.actual_TCP_pose[2])

                vx.append(state.actual_TCP_speed[0])
                vy.append(state.actual_TCP_speed[1])
                vz.append(state.actual_TCP_speed[2])

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
    num_rows = np.shape(input_q0)[0]
    control_point = np.hstack((input_q0, input_q1, input_q2, input_q3, input_q4, input_q5))
    t_blend = 0.2*np.diff(time_sequence)[0]  # time for parabolic segments
    t_final = time_sequence[-1]
    planner = LinearParabolic(time_sequence, t_blend, control_point)
    M_lq0, M_lq1, M_lq2, M_lq3, M_lq4, M_lq5 = planner.co_line()
    M_pq0, M_pq1, M_pq2, M_pq3, M_pq4, M_pq5 = planner.co_parabolic()
    M_q0 = planner.co_linear_parabolic(M_lq0, M_pq0)
    M_q1 = planner.co_linear_parabolic(M_lq1, M_pq1)
    M_q2 = planner.co_linear_parabolic(M_lq2, M_pq2)
    M_q3 = planner.co_linear_parabolic(M_lq3, M_pq3)
    M_q4 = planner.co_linear_parabolic(M_lq4, M_pq4)
    M_q5 = planner.co_linear_parabolic(M_lq5, M_pq5)
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
        print(f"dt:{t_current - t_prev}")
        # read state from the robot
        if state.runtime_state > 1 and t_current <= t_final:
            #   ----------- minimum_jerk trajectory --------------
            print('t_c=', t_current)
            for i in range(2 * num_rows - 1):
                if time_interval[i] <= t_current < time_interval[i + 1]:
                    res_q0 = planner.calc_lp(M_q0, i, t_current)
                    res_q1 = planner.calc_lp(M_q1, i, t_current)
                    res_q2 = planner.calc_lp(M_q2, i, t_current)
                    res_q3 = planner.calc_lp(M_q3, i, t_current)
                    res_q4 = planner.calc_lp(M_q4, i, t_current)
                    res_q5 = planner.calc_lp(M_q5, i, t_current)
                    break
            position_ref = [res_q0[0], res_q1[0], res_q2[0], res_q3[0], res_q4[0], res_q5[0]]
            velocity_ref = [res_q0[1], res_q1[1], res_q2[1], res_q3[1], res_q4[1], res_q5[1]]
            acceleration_ref = [res_q0[2], res_q1[2], res_q2[2], res_q3[2], res_q4[2], res_q5[2]]
            # ------------------ impedance -----------------------
            # current_joint_pose = state.actual_q
            # current_joint_speed = state.actual_qd
            # current_joint_current = state.actual_current
            # current_TCP_pose = state.actual_TCP_pose
            # current_TCP_speed = state.actual_TCP_speed
            # pose = position_ref
            print('len(pose)=', len(position_ref))
            list_to_setp(setp, position_ref)
            con.send(setp)
            if plotter:
                time_plot.append(state.timestamp - t_start)
                min_jerk_q0.append(position_ref[0] * 57.3248)
                min_jerk_q1.append(position_ref[1] * 57.3248)
                min_jerk_q2.append(position_ref[2] * 57.3248)
                min_jerk_q3.append(position_ref[3] * 57.3248)
                min_jerk_q4.append(position_ref[4] * 57.3248)
                min_jerk_q5.append(position_ref[5] * 57.3248)

                min_jerk_d_q0.append(velocity_ref[0] * 57.3248)
                min_jerk_d_q1.append(velocity_ref[1] * 57.3248)
                min_jerk_d_q2.append(velocity_ref[2] * 57.3248)
                min_jerk_d_q3.append(velocity_ref[3] * 57.3248)
                min_jerk_d_q4.append(velocity_ref[4] * 57.3248)
                min_jerk_d_q5.append(velocity_ref[5] * 57.3248)

                min_jerk_d2_q0.append(acceleration_ref[0] * 57.3248)
                min_jerk_d2_q1.append(acceleration_ref[1] * 57.3248)
                min_jerk_d2_q2.append(acceleration_ref[2] * 57.3248)
                min_jerk_d2_q3.append(acceleration_ref[3] * 57.3248)
                min_jerk_d2_q4.append(acceleration_ref[4] * 57.3248)
                min_jerk_d2_q5.append(acceleration_ref[5] * 57.3248)

                q0.append(state.actual_q[0] * 57.3248)
                q1.append(state.actual_q[1] * 57.3248)
                q2.append(state.actual_q[2] * 57.3248)
                q3.append(state.actual_q[3] * 57.3248)
                q4.append(state.actual_q[4] * 57.3248)
                q5.append(state.actual_q[5] * 57.3248)

                d_q0.append(state.actual_qd[0] * 57.3248)
                d_q1.append(state.actual_qd[1] * 57.3248)
                d_q2.append(state.actual_qd[2] * 57.3248)
                d_q3.append(state.actual_qd[3] * 57.3248)
                d_q4.append(state.actual_qd[4] * 57.3248)
                d_q5.append(state.actual_qd[5] * 57.3248)

                if len(time_plot) <= 10:
                    calc_acc_0.append(0)
                    calc_acc_1.append(0)
                    calc_acc_2.append(0)
                    calc_acc_3.append(0)
                    calc_acc_4.append(0)
                    calc_acc_5.append(0)
                else:
                    dv0 = d_q0[-1] - d_q0[-3]
                    dv1 = d_q1[-1] - d_q1[-3]
                    dv2 = d_q2[-1] - d_q2[-3]
                    dv3 = d_q3[-1] - d_q3[-3]
                    dv4 = d_q4[-1] - d_q4[-3]
                    dv5 = d_q5[-1] - d_q5[-3]
                    calc_acc_0.append(dv0 / (dt * 2))
                    calc_acc_1.append(dv1 / (dt * 2))
                    calc_acc_2.append(dv2 / (dt * 2))
                    calc_acc_3.append(dv3 / (dt * 2))
                    calc_acc_4.append(dv4 / (dt * 2))
                    calc_acc_5.append(dv5 / (dt * 2))

                px.append(state.actual_TCP_pose[0])
                py.append(state.actual_TCP_pose[1])
                pz.append(state.actual_TCP_pose[2])

                vx.append(state.actual_TCP_speed[0])
                vy.append(state.actual_TCP_speed[1])
                vz.append(state.actual_TCP_speed[2])

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
    control_point = np.hstack((input_q0, input_q1, input_q2, input_q3, input_q4, input_q5))
    ''' user define value '''
    # -----------------------user defined value---------------------------
    # here users can define  v,a,j at start and end point and knot velocity
    d_q_cp, d2_q_cp, d3_q_cp = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    # -----------------------user defined value end---------------------------
    planner = Spline535(time_sequence, control_point, d_q_cp, d2_q_cp, d3_q_cp)
    if planner.num_rows == 5:
        M_q0 = planner.co_calculate_wp2(input_q0)
        M_q1 = planner.co_calculate_wp2(input_q1)
        M_q2 = planner.co_calculate_wp2(input_q2)
        M_q3 = planner.co_calculate_wp2(input_q3)
        M_q4 = planner.co_calculate_wp2(input_q4)
        M_q5 = planner.co_calculate_wp2(input_q5)
        time_interval = planner.time_interval
    elif planner.num_rows == 4:
        M_q0 = planner.co_calculate_wp2(input_q0)
        M_q1 = planner.co_calculate_wp2(input_q1)
        M_q2 = planner.co_calculate_wp2(input_q2)
        M_q3 = planner.co_calculate_wp2(input_q3)
        M_q4 = planner.co_calculate_wp2(input_q4)
        M_q5 = planner.co_calculate_wp2(input_q5)
        time_interval = planner.time_interval
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
        if state.runtime_state > 1 and t_current <= t_final:
            #   ----------- minimum_jerk trajectory --------------
            print('t_c=', t_current)
            for i in range(3 * planner.num_rows - 3):
                if planner.time_interval[i] <= t_current < planner.time_interval[i + 1]:
                    res_q0 = planner.calc_535(M_q0[i], t_current)
                    res_q1 = planner.calc_535(M_q1[i], t_current)
                    res_q2 = planner.calc_535(M_q2[i], t_current)
                    res_q3 = planner.calc_535(M_q3[i], t_current)
                    res_q4 = planner.calc_535(M_q4[i], t_current)
                    res_q5 = planner.calc_535(M_q5[i], t_current)
                    break
            position_ref = [res_q0[0], res_q1[0], res_q2[0], res_q3[0], res_q4[0], res_q5[0]]
            velocity_ref = [res_q0[1], res_q1[1], res_q2[1], res_q3[1], res_q4[1], res_q5[1]]
            acceleration_ref = [res_q0[2], res_q1[2], res_q2[2], res_q3[2], res_q4[2], res_q5[2]]
            # ------------------ impedance -----------------------
            # current_joint_pose = state.actual_q
            # current_joint_speed = state.actual_qd
            # current_joint_current = state.actual_current  # joint current
            # current_TCP_pose = state.actual_TCP_pose
            # current_TCP_speed = state.actual_TCP_speed

            list_to_setp(setp, position_ref)
            con.send(setp)

            if plotter:
                time_plot.append(state.timestamp - t_start)
                min_jerk_q0.append(position_ref[0] * 57.3248)
                min_jerk_q1.append(position_ref[1] * 57.3248)
                min_jerk_q2.append(position_ref[2] * 57.3248)
                min_jerk_q3.append(position_ref[3] * 57.3248)
                min_jerk_q4.append(position_ref[4] * 57.3248)
                min_jerk_q5.append(position_ref[5] * 57.3248)

                min_jerk_d_q0.append(velocity_ref[0] * 57.3248)
                min_jerk_d_q1.append(velocity_ref[1] * 57.3248)
                min_jerk_d_q2.append(velocity_ref[2] * 57.3248)
                min_jerk_d_q3.append(velocity_ref[3] * 57.3248)
                min_jerk_d_q4.append(velocity_ref[4] * 57.3248)
                min_jerk_d_q5.append(velocity_ref[5] * 57.3248)

                min_jerk_d2_q0.append(acceleration_ref[0] * 57.3248)
                min_jerk_d2_q1.append(acceleration_ref[1] * 57.3248)
                min_jerk_d2_q2.append(acceleration_ref[2] * 57.3248)
                min_jerk_d2_q3.append(acceleration_ref[3] * 57.3248)
                min_jerk_d2_q4.append(acceleration_ref[4] * 57.3248)
                min_jerk_d2_q5.append(acceleration_ref[5] * 57.3248)

                q0.append(state.actual_q[0] * 57.3248)
                q1.append(state.actual_q[1] * 57.3248)
                q2.append(state.actual_q[2] * 57.3248)
                q3.append(state.actual_q[3] * 57.3248)
                q4.append(state.actual_q[4] * 57.3248)
                q5.append(state.actual_q[5] * 57.3248)

                d_q0.append(state.actual_qd[0] * 57.3248)
                d_q1.append(state.actual_qd[1] * 57.3248)
                d_q2.append(state.actual_qd[2] * 57.3248)
                d_q3.append(state.actual_qd[3] * 57.3248)
                d_q4.append(state.actual_qd[4] * 57.3248)
                d_q5.append(state.actual_qd[5] * 57.3248)

                if len(time_plot) <= 10:
                    calc_acc_0.append(0)
                    calc_acc_1.append(0)
                    calc_acc_2.append(0)
                    calc_acc_3.append(0)
                    calc_acc_4.append(0)
                    calc_acc_5.append(0)
                else:
                    dv0 = d_q0[-1] - d_q0[-3]
                    dv1 = d_q1[-1] - d_q1[-3]
                    dv2 = d_q2[-1] - d_q2[-3]
                    dv3 = d_q3[-1] - d_q3[-3]
                    dv4 = d_q4[-1] - d_q4[-3]
                    dv5 = d_q5[-1] - d_q5[-3]
                    calc_acc_0.append(dv0 / (dt * 2))
                    calc_acc_1.append(dv1 / (dt * 2))
                    calc_acc_2.append(dv2 / (dt * 2))
                    calc_acc_3.append(dv3 / (dt * 2))
                    calc_acc_4.append(dv4 / (dt * 2))
                    calc_acc_5.append(dv5 / (dt * 2))

                px.append(state.actual_TCP_pose[0])
                py.append(state.actual_TCP_pose[1])
                pz.append(state.actual_TCP_pose[2])

                vx.append(state.actual_TCP_speed[0])
                vy.append(state.actual_TCP_speed[1])
                vz.append(state.actual_TCP_speed[2])

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
    control_point = np.hstack((input_q0, input_q1, input_q2, input_q3, input_q4, input_q4))
    '''user define vel and acc at each control point for each joint'''
    # unit is degree/s 
    input_d_q = [[0,    0,      0,    0],  # j0  degree/s
                 [0,    0,      0,    0],  # j1
                 [0,    0,      0,    0],  # j2
                 [0,    0,      0,    0],  # j3
                 [0,    0,      0,    0],  # j4
                 [0,    0,      0,    0]]  # j5
    # if defined acc not all equal to 0, there will be an obvious velocity spike at the initial point

    # unit is degree/s^2 
    input_d2_q = [[0,    0,      0,    0],  # j0  degree/s^2
                  [0,    0,      0,    0],  # j1
                  [0,    0,      0,    0],  # j2
                  [0,    0,      0,    0],  # j3
                  [0,    0,      0,    0],  # j4
                  [0,    0,      0,    0]]  # j5

    # in calculation, values are calculated in rad
    input_d_q = np.array(input_d_q)/57
    input_d2_q = np.array(input_d2_q)/57

    t_final = time_sequence[-1]
    planner = QuinticBezier(time_sequence, control_point, input_d_q, input_d2_q)
    point_set_q0 = planner.two_point_to_six(input_q0, 0)
    point_set_q1 = planner.two_point_to_six(input_q1, 1)
    point_set_q2 = planner.two_point_to_six(input_q2, 2)
    point_set_q3 = planner.two_point_to_six(input_q3, 3)
    point_set_q4 = planner.two_point_to_six(input_q4, 4)
    point_set_q5 = planner.two_point_to_six(input_q5, 5)
    # ------------------- Control loop --------------------
    state = con.receive()
    tcp = state.actual_TCP_pose
    t_start = state.timestamp
    t_current = 0
    while state.timestamp - t_start < t_final:
        state = con.receive()
        t_prev = t_current
        t_current = state.timestamp - t_start
        dt = t_current - t_prev

        # read state from the robot
        if state.runtime_state > 1 and t_current <= t_final:
            # ------------- minimum_jerk trajectory --------------
            for i in range(planner.num_rows - 1):
                # time interval judgment
                if time_sequence[i] <= t_current < time_sequence[i + 1]:
                    # return pos, vel, acc
                    res_q0 = planner.calc_quintic_Bezier(i, point_set_q0[i], t_current)
                    res_q1 = planner.calc_quintic_Bezier(i, point_set_q1[i], t_current)
                    res_q2 = planner.calc_quintic_Bezier(i, point_set_q2[i], t_current)
                    res_q3 = planner.calc_quintic_Bezier(i, point_set_q3[i], t_current)
                    res_q4 = planner.calc_quintic_Bezier(i, point_set_q4[i], t_current)
                    res_q5 = planner.calc_quintic_Bezier(i, point_set_q5[i], t_current)
                    position_ref = [res_q0[0], res_q1[0], res_q2[0], res_q3[0], res_q4[0], res_q5[0]]
                    velocity_ref = [res_q0[1], res_q1[1], res_q2[1], res_q3[1], res_q4[1], res_q5[1]]
                    acceleration_ref = [res_q0[2], res_q1[2], res_q2[2], res_q3[2], res_q4[2], res_q5[2]]
                    break
            # ------------------ impedance -----------------------
            # current_joint_pose = state.actual_q
            # current_joint_speed = state.actual_qd
            # current_joint_current = state.actual_current
            # current_TCP_pose = state.actual_TCP_pose
            # current_TCP_speed = state.actual_TCP_speed
            # pose = position_ref
            #print('len(pose)=', len(position_ref))
            list_to_setp(setp, position_ref)
            con.send(setp)

            if plotter:
                time_plot.append(state.timestamp - t_start)
                min_jerk_q0.append(position_ref[0] * 57.3248)
                min_jerk_q1.append(position_ref[1] * 57.3248)
                min_jerk_q2.append(position_ref[2] * 57.3248)
                min_jerk_q3.append(position_ref[3] * 57.3248)
                min_jerk_q4.append(position_ref[4] * 57.3248)
                min_jerk_q5.append(position_ref[5] * 57.3248)

                min_jerk_d_q0.append(velocity_ref[0] * 57.3248)
                min_jerk_d_q1.append(velocity_ref[1] * 57.3248)
                min_jerk_d_q2.append(velocity_ref[2] * 57.3248)
                min_jerk_d_q3.append(velocity_ref[3] * 57.3248)
                min_jerk_d_q4.append(velocity_ref[4] * 57.3248)
                min_jerk_d_q5.append(velocity_ref[5] * 57.3248)

                min_jerk_d2_q0.append(acceleration_ref[0] * 57.3248)
                min_jerk_d2_q1.append(acceleration_ref[1] * 57.3248)
                min_jerk_d2_q2.append(acceleration_ref[2] * 57.3248)
                min_jerk_d2_q3.append(acceleration_ref[3] * 57.3248)
                min_jerk_d2_q4.append(acceleration_ref[4] * 57.3248)
                min_jerk_d2_q5.append(acceleration_ref[5] * 57.3248)

                q0.append(state.actual_q[0] * 57.3248)
                q1.append(state.actual_q[1] * 57.3248)
                q2.append(state.actual_q[2] * 57.3248)
                q3.append(state.actual_q[3] * 57.3248)
                q4.append(state.actual_q[4] * 57.3248)
                q5.append(state.actual_q[5] * 57.3248)

                d_q0.append(state.actual_qd[0] * 57.3248)
                d_q1.append(state.actual_qd[1] * 57.3248)
                d_q2.append(state.actual_qd[2] * 57.3248)
                d_q3.append(state.actual_qd[3] * 57.3248)
                d_q4.append(state.actual_qd[4] * 57.3248)
                d_q5.append(state.actual_qd[5] * 57.3248)

                if len(time_plot) <= 10:
                    calc_acc_0.append(0)
                    calc_acc_1.append(0)
                    calc_acc_2.append(0)
                    calc_acc_3.append(0)
                    calc_acc_4.append(0)
                    calc_acc_5.append(0)
                else:
                    dv0 = d_q0[-1] - d_q0[-3]
                    dv1 = d_q1[-1] - d_q1[-3]
                    dv2 = d_q2[-1] - d_q2[-3]
                    dv3 = d_q3[-1] - d_q3[-3]
                    dv4 = d_q4[-1] - d_q4[-3]
                    dv5 = d_q5[-1] - d_q5[-3]
                    calc_acc_0.append(dv0 / (dt * 2))
                    calc_acc_1.append(dv1 / (dt * 2))
                    calc_acc_2.append(dv2 / (dt * 2))
                    calc_acc_3.append(dv3 / (dt * 2))
                    calc_acc_4.append(dv4 / (dt * 2))
                    calc_acc_5.append(dv5 / (dt * 2))

                px.append(state.actual_TCP_pose[0])
                py.append(state.actual_TCP_pose[1])
                pz.append(state.actual_TCP_pose[2])

                vx.append(state.actual_TCP_speed[0])
                vy.append(state.actual_TCP_speed[1])
                vz.append(state.actual_TCP_speed[2])

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
save_npy = 1
if save_npy:
    # Step 1: Define the folder path
    folder_path = r'C:\Users\Kerouac\Desktop\project_UR3e\Servoj_RTDE_UR3-main\joint_space_' + run_mode  # You can change this to any folder name or path you prefer

    # Step 2: Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # print(f"Saving to file: {run_mode+' TCP_position_3D_robot.npy'}")
    np.save(os.path.join(folder_path, run_mode + ' TCP_position_3D_robot.npy'), [px, py, pz])

    # print(f"Saving to file: {run_mode+' joint_position_6_joints_robot.npy'}")
    np.save(os.path.join(folder_path, run_mode + ' position_6_joints_robot.npy'), [q0, q1, q2, q3, q4, q5])

    # print(f"Saving to file: {run_mode+' joint_velocity_6_joints_robot.npy'}")
    np.save(os.path.join(folder_path, run_mode+' velocity_6_joints_robot'), [d_q0, d_q1, d_q2, d_q3, d_q4, d_q5])

    # print(f"Saving to file: {run_mode+' joint_acceleration_6_joints_planned.npy'}")
    np.save(os.path.join(folder_path, run_mode+' acceleration_6_joints_robot'),
            [calc_acc_0, calc_acc_1, calc_acc_2, calc_acc_3, calc_acc_4, calc_acc_5])

    # print(f"Saving to file: {run_mode+' joint_acceleration_6_joints_planned.npy'}")
    np.save(os.path.join(folder_path, run_mode+' acceleration_6_joints_planned'),
            [min_jerk_d2_q0, min_jerk_d2_q1, min_jerk_d2_q2, min_jerk_d2_q3, min_jerk_d2_q4, min_jerk_d2_q5])

    # print(f"Saving to file: {run_mode+' joint_space_measured_current.npy'}")
    np.save(os.path.join(folder_path, run_mode+' measured_current'), [joint_current_0, joint_current_1, joint_current_2, joint_current_3, joint_current_4, joint_current_5])

    # print(f"Saving to file: {run_mode + ' joint_space_power.npy'}")
    np.save(os.path.join(folder_path, run_mode + ' power'),
            [joint_power_0, joint_power_1, joint_power_2, joint_power_3, joint_power_4, joint_power_5])

    # print(f"Saving to file: {run_mode+' joint_space_consumed_energy.npy'}")
    np.save(os.path.join(folder_path, run_mode+' consumed energy'), joint_energy)

    print(f'{run_mode}: Data saved successfully')

'''plot'''
plot_auto_save = 1
if plotter:
    input_q0 = input_q0 / 0.01745
    input_q1 = input_q1 / 0.01745
    input_q2 = input_q2 / 0.01745
    input_q3 = input_q3 / 0.01745
    input_q4 = input_q4 / 0.01745
    input_q5 = input_q5 / 0.01745
    # ----------- joint position -------------
    plt.figure(0)
    plt.title('position_q0 '+run_mode)
    plt.plot(time_plot, q0, label="q0_robot")
    plt.plot(time_plot, min_jerk_q0, label="q0_planned")
    plt.plot(time_sequence, input_q0, 'o', label='q0')
    plt.legend()
    plt.grid()
    plt.ylabel('Position in joint0[°]')
    plt.xlabel('Time [sec]')

    plt.figure(1)
    plt.title('position_q1 '+run_mode)
    plt.plot(time_plot, q1, label="q1_robot")
    plt.plot(time_plot, min_jerk_q1, label="q1_planned")
    plt.plot(time_sequence, input_q1, 'o', label='q1')
    plt.legend()
    plt.grid()
    plt.ylabel('Position in joint1[°]')
    plt.xlabel('Time [sec]')

    plt.figure(2)
    plt.title('position_q2 '+run_mode)
    plt.plot(time_plot, q2, label="q2_robot")
    plt.plot(time_plot, min_jerk_q2, label="q2_planned")
    plt.plot(time_sequence, input_q2, 'o', label='q2')
    plt.legend()
    plt.grid()
    plt.ylabel('Position in joint2[°]')
    plt.xlabel('Time [sec]')

    plt.figure(3)
    plt.title('position_q3 '+run_mode)
    plt.plot(time_plot, q3, label="q3_robot")
    plt.plot(time_plot, min_jerk_q3, label="q3_planned")
    plt.plot(time_sequence, input_q3, 'o', label='q3')
    plt.legend()
    plt.grid()
    plt.ylabel('Position in joint3[°]')
    plt.xlabel('Time [sec]')

    plt.figure(4)
    plt.title('position_q4 '+run_mode)
    plt.plot(time_plot, q4, label="q4_robot")
    plt.plot(time_plot, min_jerk_q4, label="q4_planned")
    plt.plot(time_sequence, input_q4, 'o', label='q4')
    plt.legend()
    plt.grid()
    plt.ylabel('Position in joint4[°]')
    plt.xlabel('Time [sec]')

    plt.figure(5)
    plt.title('position_q5 '+run_mode)
    plt.plot(time_plot, q5, label="q5_robot")
    plt.plot(time_plot, min_jerk_q5, label="q5_planned")
    plt.plot(time_sequence, input_q5, 'o', label='q5')
    plt.legend()
    plt.grid()
    plt.ylabel('Position in joint5[°]')
    plt.xlabel('Time [sec]')
    # ----------- velocity -------------
    plt.figure(6)
    plt.title('d_q0 '+run_mode)
    plt.plot(time_plot, d_q0, label=" d_q0_robot")
    plt.plot(time_plot, min_jerk_d_q0, label=" d_q0_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Angle Vel [°/s]')
    plt.xlabel('Time [sec]')

    plt.figure(7)
    plt.title('d_q1 '+run_mode)
    plt.plot(time_plot, d_q1, label=" d_q1_robot")
    plt.plot(time_plot, min_jerk_d_q1, label=" d_q1_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Angle Vel [°/s]')
    plt.xlabel('Time [sec]')

    plt.figure(8)
    plt.title('d_q2 '+run_mode)
    plt.plot(time_plot, d_q2, label="d_q2_robot")
    plt.plot(time_plot, min_jerk_d_q2, label="d_q2_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Angle Vel [°/s]')
    plt.xlabel('Time [sec]')

    plt.figure(9)
    plt.title('d_q3 '+run_mode)
    plt.plot(time_plot, d_q3, label=" d_q3_robot")
    plt.plot(time_plot, min_jerk_d_q3, label=" d_q3_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Angle Vel [°/s]')
    plt.xlabel('Time [sec]')

    plt.figure(10)
    plt.title('d_q4 '+run_mode)
    plt.plot(time_plot, d_q4, label=" d_q4_robot")
    plt.plot(time_plot, min_jerk_d_q4, label=" d_q4_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Angle Vel [°/s]')
    plt.xlabel('Time [sec]')

    plt.figure(11)
    plt.title('d_q5 '+run_mode)
    plt.plot(time_plot, d_q5, label="d_q5_robot")
    plt.plot(time_plot, min_jerk_d_q5, label="d_q5_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Angle Vel [°/s]')
    plt.xlabel('Time [sec]')
    # ----------- Acceleration -------------
    plt.figure(12)
    plt.title('Acceleration '+run_mode)
    plt.plot(time_plot, calc_acc_0, label='acc0 = dv/dt')
    plt.plot(time_plot, min_jerk_d2_q0, label="acc_j0_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Acceleration [°/s^2]')
    plt.xlabel('Time [sec]')

    plt.figure(13)
    plt.title('Acceleration '+run_mode)
    plt.plot(time_plot, calc_acc_1, label='acc1 = dv/dt')
    plt.plot(time_plot, min_jerk_d2_q1, label="acc_j1_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Acceleration [°/s^2]')
    plt.xlabel('Time [sec]')

    plt.figure(14)
    plt.title('Acceleration '+run_mode)
    plt.plot(time_plot, calc_acc_2, label='acc2 = dv/dt')
    plt.plot(time_plot, min_jerk_d2_q2, label="acc_j2_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Acceleration [°/s^2]')
    plt.xlabel('Time [sec]')

    plt.figure(15)
    plt.title('Acceleration '+run_mode)
    plt.plot(time_plot, calc_acc_3, label='acc3 = dv/dt')
    plt.plot(time_plot, min_jerk_d2_q3, label="acc_j3_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Acceleration [°/s^2]')
    plt.xlabel('Time [sec]')

    plt.figure(16)
    plt.title('Acceleration '+run_mode)
    plt.plot(time_plot, calc_acc_4, label='acc4 = dv/dt')
    plt.plot(time_plot, min_jerk_d2_q4, label="acc_j4_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Acceleration [°/s^2]')
    plt.xlabel('Time [sec]')

    plt.figure(17)
    plt.title('Acceleration '+run_mode)
    plt.plot(time_plot, calc_acc_5, label='acc5 = dv/dt')
    plt.plot(time_plot, min_jerk_d2_q5, label="acc_j5_planned")
    plt.legend()
    plt.grid()
    plt.ylabel('Acceleration [°/s^2]')
    plt.xlabel('Time [sec]')

    # ----------- task space visualization -------------
    fig = plt.figure(19)
    plt.title('TCP_3D '+run_mode)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(input_x[0], input_y[0], input_z[0], color='red', label='start point')
    ax.scatter3D(input_x[-1], input_y[-1], input_z[-1], color='blue', label='desired point')
    ax.scatter3D(input_x[1:-1], input_y[1:-1], input_z[1:-1], color='green', label='via point')
    ax.plot3D(px, py, pz, color='green', label='actual trajectory')
    ax.set_xlabel('TCP POS X position [m]')
    ax.set_ylabel('TCP POS Y position [m]')
    ax.set_zlabel('TCP POS Z position [m]')
    ax.legend(), plt.grid(True)
    plt.tight_layout()

    plt.figure(20)
    plt.title('TCP position_xyz '+run_mode)
    plt.plot(time_plot, px, label="TCP x_robot")
    plt.plot(time_sequence, input_x, 'o', label='control point x coordinate')
    plt.plot(time_plot, py, label="TCP y_robot")
    plt.plot(time_sequence, input_y, 'o', label='control point y coordinate')
    plt.plot(time_plot, pz, label="TCP z_robot")
    plt.plot(time_sequence, input_z, 'o', label='control point z coordinate')
    plt.legend(), plt.grid()
    plt.ylabel('Position in x-y-z[m]')
    plt.xlabel('Time [sec]')
    # ----------- TCP velocity -------------
    plt.figure(21)
    plt.title('TCP velocity_x '+run_mode)
    plt.plot(time_plot, vx, label="TCP vx_robot")
    plt.legend()
    plt.grid()
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time [sec]')

    plt.figure(22)
    plt.title('TCP velocity_y '+run_mode)
    plt.plot(time_plot, vy, label="TCP vy_robot")
    plt.legend()
    plt.grid()
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time [sec]')

    plt.figure(23)
    plt.title('TCP velocity_z '+run_mode)
    plt.plot(time_plot, vz, label="TCP vz_robot")
    plt.legend()
    plt.grid()
    plt.ylabel('Velocity [m/s]')
    plt.xlabel('Time [sec]')
    #
    # # ----------- Current -------------
    plt.figure(24)
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
    plt.figure(25)
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
    plt.figure(26)
    plt.title(run_mode + ' joint_energy')
    categories = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
    plt.bar(categories, joint_energy, label='consumed energy')
    plt.legend()
    plt.grid()
    plt.xlabel('joints')
    plt.ylabel('consumed energy')
    # ----------- auto save -------------
    if plot_auto_save:
        output_dir = r'C:\Users\Kerouac\Desktop\plot_joint_space_' + run_mode
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)  # create if not exist

        # acquire the plot_num
        figures = plt.get_fignums()  # 返回所有图形编号的列表
        # 遍历所有生成的图形并保存
        for i, fig_num in enumerate(figures):
            fig = plt.figure(fig_num)  # 获取当前编号的图形
            # 保存图像为svg
            file_path = os.path.join(output_dir, f'_joint_space_{run_mode}_plot_{i}.svg')  # 生成文件名
            fig.savefig(file_path, format='svg')
            # save as png
            file_path = os.path.join(output_dir, f'_joint_space_{run_mode}_plot_{i}.png')
            fig.savefig(file_path, format='png')
            # print(f"Plot {i + 1} saved to {file_path}")
        print("All plots have been saved.")

    plt.show()








