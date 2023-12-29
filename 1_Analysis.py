import pybullet as p
import time
import numpy as np
import pybullet_data
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize

dt = 1/240 # pybullet simulation step
q0 = 0.5   # starting position (radian)
physicsClient = p.connect(p.GUI) # or p.DIRECT for non-graphical version (GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("simple.urdf", useFixedBase=True)

# get rid of all the default damping forces
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# go to the starting position
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetPosition=q0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)
t, theta, omega = [0], np.array([q0]), np.array([0])
for i in range (2400):
    p.stepSimulation()
    time.sleep(20000)
    #time.sleep(dt)
    theta = np.append(theta, p.getJointState(boxId, 1)[0])
    omega = np.append(omega, p.getJointState(boxId, 1)[1])  # Не понадобилась
    t.append(i * dt)
p.disconnect()

# Уравнение маятника
def f(y, x):
    b, m, l, g = 1, 1, 0.8, 10
    return np.array([y[1], -b/(m*l**2) * y[1] - g/l*np.sin(y[0])])

# Метод Эйлера
def euler(end):
    h = dt
    x, y = [0], [y0]
    while x[-1] < end - h:
        x.append(x[-1] + h)
        y.append([0, y[-1][1] + h * f(y[-1], x[-1])[1]])
        y[-1][0] = (y[-2][0] + h * f([y[-2][0], y[-1][1]], x[-1])[0])
    h = end - x[-1]
    x.append(end)
    y.append([0, y[-1][1] + h * f(y[-1], x[-1])[1]])
    y[-1][0] = (y[-2][0] + h * f([y[-2][0], y[-1][1]], x[-1])[0])
    return x, y

y0 = [q0, 0]    # Начальное условие

y1 = odeint(f, y0, t)
y1 = np.array(list(map(lambda i: y1[i][0], range(len(y1)))))    # Нас интересует только угол отклонения

[x2, y2] = euler(t[-1])
y2 = np.array(list(map(lambda i: y2[i][0], range(len(y2)))))    # Нас интересует только угол отклонения

aver_1 = sum(abs(theta - y1))/len(y1)
max_1 = max(abs(theta - y1))
aver_2 = sum(abs(theta - y2))/len(y2)
max_2 = max(abs(theta - y2))

print ('Погрешность odeint:', '\n', 'Средняя:', round(aver_1, 7), '\n', 'Максимальная:', round(max_1, 7), '\n')
print ('Погрешность Эйлера:', '\n', 'Средняя:', round(aver_2, 16), '\n', 'Максимальная:', round(max_2, 16), '\n')

plt.figure('Analysis')
plt.title('Положение маятника')
plt.xlabel('Время')
plt.ylabel('Угол')
plt.plot(t, theta, t, y1, x2, y2, 'y')
plt.legend(['Исходный', 'odeint', 'Метод Эйлера'])
plt.axhline(y = 0, color='gray')
plt.axvline(x = 0, color='gray')
plt.show()

# Уравнение маятника с неизвестными коэффициэнтами
def fun_search(y, k):   
    return np.array([y[1], -k[0] * y[1] - k[1] * np.sin(y[0])])

def euler_search(end, k):   # Отдельный метод Эйлера, т.к иначе не придумал как подать k в функцию
    h = dt
    x, y = [0], [y0]
    while x[-1] < end - h:
        x.append(x[-1] + h)
        y.append([0, y[-1][1] + h * fun_search(y[-1], k)[1]])
        y[-1][0] = (y[-2][0] + h * fun_search([y[-2][0], y[-1][1]], k)[0])
    h = end - x[-1]
    x.append(end)
    y.append([0, y[-1][1] + h * fun_search(y[-1], k)[1]])
    y[-1][0] = (y[-2][0] + h * fun_search([y[-2][0], y[-1][1]], k)[0])
    y = np.array(list(map(lambda i: y[i][0], range(len(y)))))
    return y

def error_search(k):
    theta_search = euler_search(t[-1], k)
    return sum(list(map(lambda i: (theta[i] - theta_search[i])**2, range(len(theta)))))

res = minimize(error_search, x0 = [1, 1])
y3 = euler_search(t[-1], res.x)

aver_3 = sum(abs(theta - y3))/len(y3)
max_3 = max(abs(theta - y3))

print ('Погрешность идентификации:', '\n', 'Средняя:', round(aver_3, 16), '\n', 'Максимальная:', round(max_3, 16))

plt.figure('Analysis')
plt.title('Положение маятника')
plt.xlabel('Время')
plt.ylabel('Угол')
plt.plot(t, theta, t, y3, 'y')
plt.legend(['Исходный', 'Полученный идентификацией'])
plt.axhline(y = 0, color='gray')
plt.axvline(x = 0, color='gray')
plt.show()

print('Полученные коэффиценты уравнения маятника:', res.x)
