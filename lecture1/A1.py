# STATEMENT OF WORK
# -ChatGPT, Github Copilot
# -Vikram, Serena, Third student whose name I forgot
# -Python, scipy optimization library (used to calculate the values)


import numpy as np
import matplotlib.pyplot as plt

# Function definition
def twoParamSig(x, params):
    b, x0 = params
    return 1 / (1 + np.exp(-b * (x - x0)))

def calculate_current(params):
    # Model parameters
    ba = params[0]  # steady state channels open per mV
    v0a = params[1]   # mV
    taua = params[2]   # msec

    bi = params[3]  # steady state channels open per mV
    v0i = params[4]   # mV
    taui = params[5]   # msec

    gBar = params[6]  # mS
    Er = params[7]    # mV

    int_count = len(data['vStep'][0])
    time_count = len(data['t'])
    xa = [[0 for x in range(time_count)] for y in range(int_count)]
    g = [[0 for x in range(time_count)] for y in range(int_count)]
    i = [[0 for x in range(time_count)] for y in range(int_count)]
    xi = [[0 for x in range(time_count)] for y in range(int_count)]

    dt = data['t'][1] - data['t'][0]
    for vSI in range(int_count):    
        v = [x[vSI] for x in data['vStep']]

        for j in range(len(data['t'])):
            t = data['t'][j]
            xaInf = twoParamSig(v[j], [ba, v0a])
            xiInf = 1 - twoParamSig(v[j], [bi, v0i])
            if j > 0:
                xa[vSI][j] = xa[vSI][j-1] + (xaInf - xa[vSI][j-1]) * (1 - np.exp(-dt/taua))
                xi[vSI][j] = xi[vSI][j-1] + (xiInf - xi[vSI][j-1]) * (1 - np.exp(-dt/taui))
            else:
                xa[vSI][j] = xaInf
                xi[vSI][j] = xiInf
            g[vSI][j] = gBar * xa[vSI][j] * xi[vSI][j]
            i[vSI][j] = g[vSI][j] * (v[j] - Er)

    return i

def calculate_error(params):
    total_error = 0
    i = calculate_current(params)
    for vSI in range(int_count):
        current = [x[vSI] for x in data['iUnknownCurrent']]
        # calculate cross-entropy loss
        # total_error += np.sum(-current * np.log(i[vSI]) - (1 - current) * np.log(1 - i[vSI]))
        for time_step in range(time_count):
            total_error += 0.5 * (current[time_step] - i[vSI][time_step]) ** 2
    return total_error

def plot_current(params):
    i = calculate_current(params)
    for vSI in range(int_count):
        color = colors[vSI % len(colors)]  # Cycle through colors
        plt.plot(data['t'], i[vSI], color=color, label=f'V step {vSI} (mV)')
    plt.ylabel('Current (mA)')
    plt.xlabel('Time')  # Assuming data['t'] represents time
    plt.grid(True)
    plt.legend(bbox_to_anchor=(0, -0.1), loc='best', borderaxespad=0.)  # Add a legend to distinguish different plots
    # move the legend outside and below the plot
    plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()


from mat4py import loadmat
mat = loadmat('lecture1/CookAssignemnt1UnknownCurrent.mat')

# each element of mat['t'] is a list of length 1, so we need to extract the value
mat['t'] = [x[0] for x in mat['t']]

data = mat


ba = 0.5
v0a = -45
gBar = .002
Er = 10
taua = 2

bi = 0.5
v0i = -45
taui = 0.5


# Plot steady state curves
t_ssac = np.array(data['t'])
plt.plot(t_ssac, twoParamSig(t_ssac, [ba, v0a]))
plt.plot(t_ssac, 1 - twoParamSig(t_ssac, [ba, v0a]))
plt.xlabel('mV')
plt.legend(['xa(t = inf)', 'xi(t = inf)'], loc='best')
plt.grid(True)
plt.figure(2)
del t_ssac

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # List of colors for different plots

int_count = len(data['vStep'][0])
time_count = len(data['t'])
xa = [[0 for x in range(time_count)] for y in range(int_count)]
g = [[0 for x in range(time_count)] for y in range(int_count)]
i = [[0 for x in range(time_count)] for y in range(int_count)]
xi = [[0 for x in range(time_count)] for y in range(int_count)]

dt = data['t'][1] - data['t'][0]

for vSI in range(int_count):    
    v = [x[vSI] for x in data['vStep']]

    for j in range(len(data['t'])):
        t = data['t'][j]
        xaInf = twoParamSig(v[j], [ba, v0a])
        xiInf = 1 - twoParamSig(v[j], [bi, v0i])
        if j > 0:
            xa[vSI][j] = xa[vSI][j-1] + (xaInf - xa[vSI][j-1]) * (1 - np.exp(-dt/taua))
            xi[vSI][j] = xi[vSI][j-1] + (xiInf - xi[vSI][j-1]) * (1 - np.exp(-dt/taui))
        else:
            xa[vSI][j] = xaInf
            xi[vSI][j] = xiInf
        g[vSI][j] = gBar * xa[vSI][j] * xi[vSI][j]
        i[vSI][j] = g[vSI][j] * (v[j] - Er) 

    color = colors[vSI % len(colors)]  # Cycle through colors
    plt.plot(data['t'], v, color=color, label=f'V step {vSI} (mV)')
    
    
plt.ylabel('V step (mV)')
plt.xlabel('Time')  # Assuming data['t'] represents time
plt.grid(True)
plt.legend()  # Add a legend to distinguish different plots
# move the legend outside and below the plot
plt.legend(bbox_to_anchor=(0, -0.1), loc='best', borderaxespad=0.)
plt.tight_layout()



# # Additional code for plotting xa, xi, g, and i can be added similarly

# Plot the current traces
plt.figure(3)

for vSI in range(int_count):
    color = colors[vSI % len(colors)]  # Cycle through colors
    plt.plot(data['t'], i[vSI], color=color, label=f'V step {vSI} (mV)')

plt.ylabel('Current (mA)')
plt.xlabel('Time')  # Assuming data['t'] represents time
plt.grid(True)
plt.legend(bbox_to_anchor=(0, -0.1), loc='best', borderaxespad=0.)  # Add a legend to distinguish different plots
# move the legend outside and below the plot
plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0.)
plt.tight_layout()

# plot the current data
plt.figure(4)
for vSI in range(int_count):
    color = colors[vSI % len(colors)]  # Cycle through colors
    current = [x[vSI] for x in data['iUnknownCurrent']]
    plt.plot(data['t'], current, color=color, label=f'V step {vSI} (mV)')
plt.ylabel('Current (mA)')
plt.xlabel('Time')  # Assuming data['t'] represents time
plt.grid(True)
plt.legend(bbox_to_anchor=(0, -0.1), loc='best', borderaxespad=0.)  # Add a legend to distinguish different plots
# move the legend outside and below the plot
plt.legend(bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.show()

total_error = 0

ba = 0.5
v0a = -45
taua = 2

gBar = .002
Er = 10

bi = 0.5
v0i = -45
taui = 0.5

params = [ba, v0a, taua, bi, v0i, taui, gBar, Er]
current = calculate_current(params)

x = np.array(current)
x.flatten()
y = np.array(data['iUnknownCurrent'])
y.flatten()

with open('lecture1/params1.txt', 'a') as f:
    from scipy.optimize import minimize
    results = minimize(fun=calculate_error, x0=params, method='Nelder-Mead')
    if results.success:
        print(results.x)
        f.write(str(results.x))

    results = minimize(fun=calculate_error, x0=params, method='Powell')
    if results.success:
        print(results.x)
        f.write(str(results.x))
        
    results = minimize(fun=calculate_error, x0=params, method='CG')
    if results.success:
        print(results.x)
        f.write(str(results.x))
    results = minimize(fun=calculate_error, x0=params, method='BFGS')
    if results.success:
        print(results.x)
        f.write(str(results.x))
    results = minimize(fun=calculate_error, x0=params, method='L-BFGS-B')
    if results.success:
        print(results.x)
        f.write(str(results.x))
    results = minimize(fun=calculate_error, x0=params, method='TNC')
    if results.success:
        print(results.x)
        f.write(str(results.x))
    results = minimize(fun=calculate_error, x0=params, method='COBYLA')
    if results.success:
        print(results.x)
        f.write(str(results.x))
    results = minimize(fun=calculate_error, x0=params, method='SLSQP')
    if results.success:
        print(results.x)
        f.write(str(results.x))
    results = minimize(fun=calculate_error, x0=params, method='trust-constr')
    if results.success:
        print(results.x)
        f.write(str(results.x))
