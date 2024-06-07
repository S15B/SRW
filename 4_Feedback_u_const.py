import pybullet as p
import time
import numpy as np
import pybullet_data
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

target_angle = np.pi/4   # нужный угол
start_pos = np.pi
dt = 1/240      # pybullet simulation step    
physicsClient = p.connect(p.DIRECT) # or p.GUI for graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("simple.urdf", useFixedBase=True)

# get rid of all the default damping forces
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# Модель уравнения маятника
b, m, l, g = 1, 1, 0.8, 10
def f(y, x, tau):
    return np.array([y[1], - b * y[1] - m*g*l * np.sin(y[0]) + tau])

# go to the starting position
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=start_pos, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

T = 2400    
tau = 0

t, acceler = [0], np.array([])
theta, omega = np.array([p.getJointState(boxId, 1)[0]]), np.array([p.getJointState(boxId, 1)[1]])
for i in range (T + 1):
    #time.sleep(dt)
    u = 0.1
    tau = b * omega[-1] + m*g*l * np.sin(theta[-1]) + (m*l**2) * u
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, controlMode=p.TORQUE_CONTROL, force=tau)
    p.stepSimulation()
    theta = np.append(theta, p.getJointState(boxId, 1)[0])
    omega = np.append(omega, p.getJointState(boxId, 1)[1])
    acceler = np.append(acceler, (omega[-1] - omega[-2])/dt)
    t.append(i * dt)
p.disconnect()


# Метод Эйлера c tau
def euler(t):
    h = dt
    u = 0.1
    tau = 0
    y = [[start_pos, 0]]
    y13 = []
    for i in t[1:]:
        tau = b * y[-1][1] + m*g*l * np.sin(y[-1][0]) + u
        y.append([0, y[-1][1] + h * f(y[-1], i, tau)[1]])
        y[-1][0] = (y[-2][0] + h * f([y[-2][0], y[-1][1]], i, tau)[0]) 
        y13.append((y[-1][1] - y[-2][1])/dt)
    return y, y13

[y1, y13] = euler(t)
y11 = np.array(list(map(lambda i: y1[i][0], range(len(y1))))) 
y12 = np.array(list(map(lambda i: y1[i][1], range(len(y1))))) 

# odeint c tau
y21, y22 = [start_pos], [0]
for i in range(len(t) - 1):
    tau = b * y22[-1] + m*g*l * np.sin(y21[-1]) + u
    y2 = odeint(f, [y21[-1], y22[-1]], [t[i], t[i + 1]], args=(tau,))
    y21.append(y2[-1][0])   # Угол
    y22.append(y2[-1][1])   # Скорость

plt.figure('Линеаризация с постоянным управлением')
plt.subplot(3, 1, 1)
plt.grid(True)
plt.title('u = {0}'.format(u))
plt.xlabel('Время')
plt.ylabel('\u03B8')
plt.plot(t, theta, t, y11)
plt.legend(['Pybullet', 'Эйлер'])
plt.subplot(3, 1, 2)
plt.grid(True)
plt.xlabel('Время')
plt.ylabel('\u03B8\'')
plt.plot(t, omega, t, y12)
plt.legend(['Pybullet', 'Эйлер'])
plt.subplot(3, 1, 3)
plt.grid(True)
plt.xlabel('Время')
plt.ylabel('\u03B8\'\'')
plt.plot(t[1:], np.round(acceler, 5), t[1:], np.round(y13, 5))
plt.legend(['Pybullet', 'Эйлер'])
plt.show()

