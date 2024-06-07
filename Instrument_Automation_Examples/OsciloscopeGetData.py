import pyvisa as pv
import matplotlib.pyplot as plt

# Use pyvisa-py backend to access the resources
rm = pv.ResourceManager('@py')

# gets the list of connected resources
resources_list = rm.list_resources()
# saves the first resource listed
instr_visa_address = resources_list[0]
print('Resource instr_visa_address: ', instr_visa_address)

# Open communication with the instrument (osciloscope) and print the information
scope = rm.open_resource(instr_visa_address)
print('Instrument info:', scope.query('*IDN?'))

# scope parameters
vRange = 4 # Voltage range on the scope screen (vertical axis range)
tRange = 50e-6 # Time range on the scope screen (horizontal axis)
# If the signal is a sine wave with frequency f=10^5 Hz, it will fit 
# 5 wave paeriods in the scope screen.
trigLevel = 500e-9
ch = 1
chImpedance = 'fifty'

#wave generator parameters
wgen_func = 'sin'
wgen_f = 1e5
wgen_v = 4
wgen_v_offs = 0

# reset the scope and waits for previous operations to finish
scope.write('*rst')
scope.write('*opc?')

# passing parameters with SCPI commands

# wave generator (internal)
scope.write(f'wgen:function {wgen_func}')
scope.write(f'wgen:freq {wgen_f}')
scope.write(f'wgen:voltage {wgen_v}')
scope.write(f'wgen:voltage:offset {wgen_v_offs}')
scope.write(f'wgen:output on')

#SCOPE
# setup up vertical and horizontal ranges (amplitude and time axis)
scope.write(f'channel{ch}:range {vRange}') # CHANnel<N>:RANGe <range_value>
scope.write(f'timebase:range {tRange}') # “TIMebase:RANGe <full_scale_range>
# setup up trigger mode and level
scope.write(f'trigger:mode edge') # TRIGger:MODE <mode>
scope.write(f'trigger:level channel{ch}, {trigLevel}') # TRIGger:LEVel CHANnel<N>, <level>
scope.write(f'channel{ch}:impedance {chImpedance}')
# SCPI commands are NOT CASE SENSITIVE!!!

# set the wave source
scope.write(f'waveform:source channel{ch}')

# Specify the saveform format
scope.write('waveform:format byte')

# capture data
scope.write('digitize')

# transfer binary waveform data from scope
data = scope.query_binary_values('waveform:data?', datatype='B') 
# Instrument binary data is unsigned (B). In case of signed binary data, 
# use datatype='b'.

# get the instrument parameter to scale the data correctly
xIncrement = float(scope.query('waveform:xincrement?'))
xOrigin = float(scope.query('waveform:xorigin?'))
yIncrement = float(scope.query('waveform:yincrement?'))
yOrigin = float(scope.query('waveform:yorigin?'))
yReference = float(scope.query('waveform:yreference?'))
length = len(data)

# applying scaling factors
# standard syntax

time = []
wfm = []

# First, construct the X axis (it goes from -xOrigin to +xOrigin)
	
time = [(t*xIncrement) + xOrigin for t in range(length)]
wfm = [((d - yReference)*yIncrement) + yOrigin for d in data]

# plot waveform data

plt.plot(time, wfm)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (v)')
plt.show()

scope.close()
rm.close()