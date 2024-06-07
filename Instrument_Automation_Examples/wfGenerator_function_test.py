import pyvisa as pv
import time

# Use pyvisa-py backend to access the resources
rm = pv.ResourceManager('@py')

# gets the ist of connected resources
resources_list = rm.list_resources()
# stores the first resource listed
instr_visa_address = resources_list[0]
print('Resource instr_visa_address: ', instr_visa_address)

wfgen = rm.open_resource(instr_visa_address)
print('Instrument info:', wfgen.query('*IDN?'))

wfgen.write('*rst') # resets the instrument to default configuration
wfgen.write('*opc?') # waits until previous commands are finished

# Parameters of a sine wave
function = 'FUNC SIN'
freq = 'FREQ +1E+05'
volt_high = 'VOLT:HIGH +2.0'
volt_low = 'VOLT:LOW 0'
phase = 'PHASE 0'

command_list = [function, freq, volt_high, volt_low, phase, 'OUTP ON']
for command in command_list:
	wfgen.write(command)

time.sleep(3)

# Parameters of a square wave
function = 'FUNC SQU'
freq = 'FREQ +1E+05'
dutC = 'FUNC:SQU:DCYC +20'
volt_high = 'VOLT:HIGH +2.0'
volt_low = 'VOLT:LOW 0'

command_list = [function, freq, dutC, volt_low, volt_high]
for command in command_list:
	wfgen.write(command)

time.sleep(3)

# Parameters of a ramp wave
function = 'FUNC RAMP'
freq = 'FREQ +1E+05'
symm = 'FUNC:RAMP:SYMM 25'
volt = 'VOLT +2.0'
offset = 'VOLT:OFFS +1'

command_list = [function, freq, symm, volt, offset]
for command in command_list:
	wfgen.write(command)

time.sleep(3)

wfgen.write('OUTP OFF')

wfgen.close()
rm.close()

print('end of program.')