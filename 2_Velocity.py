import pybullet as p
import time
import numpy as np
import pybullet_data
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

dt = 1/240      # pybullet simulation step 
target_angle = np.pi/4   
physicsClient = p.connect(p.DIRECT) # or p.DIRECT for non-graphical version (GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("simple.urdf", useFixedBase=True)

# get rid of all the default damping forces
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

t, theta, omega = [0], np.array([0]), np.array([0])
k = 0
for i in range (360):
    p.stepSimulation()
    speed = 8 * (target_angle - theta[-1])
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=speed, controlMode=p.VELOCITY_CONTROL)
    theta = np.append(theta, p.getJointState(boxId, 1)[0])
    omega = np.append(omega, p.getJointState(boxId, 1)[1])
    t.append(i * dt)
    if abs(theta[-1] - target_angle) <= 10**(-3) and k == 0:
        print ('Дошли за {0} секунд'.format(round(t[-1], 5)))
        k = 1
p.disconnect()


plt.figure('VELOCITY_CONTROL')
plt.title('Приведение в положение {0}'.format(round(target_angle, 4)))
plt.xlabel('Время')
plt.ylabel('')
plt.plot(t, theta, t, omega)
plt.axhline(y = target_angle, color='yellow', linestyle='dashed')
plt.legend(['Положение', 'Скорость', 'Нужный угол'])
plt.axhline(y = 0, color='gray')
plt.axhline(y = target_angle, color='yellow', linestyle='dashed')
plt.axvline(x = 0, color='gray')
plt.show()

