# UR3e
Trajectory generator, RTDE and URScript 

UR3e is a robot arm from Universal Robot.
The frame to realize RTDE and the original URScript are from the Internet and not created by me.
Trajectory generator includes 5 algorithms, fosucing on joint space or task space. 
5 algorithms are written as 'class', which makes it convient to call in other files.

RTDE is embedded in the two files starts with 'min_jerk_xxx.py'. The 'min_jerk_xxx.py' files are to conduct the codes and communicate with the virtual and real robot.
The whole thing goes like this: 
  First users run the 'min_jerk_..._joint.py' or 'min_jerk_..._task.py'. Trajectory are genearting, waiting for data transfer to the robot.
  Then the it triggers the 'URScrpit.urp' running in the simulator (usually named as URSim) in the virtual machine. Data are received.
  The code in 'min_jerk_..._joint.py' and 'min_jerk_..._space.py'must correspond with 'URScrpit.urp'.
