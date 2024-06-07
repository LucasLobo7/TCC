import pyvisa as pv

# Use pyvisa-py backend to access the resources
rm = pv.ResourceManager('@py')

# gets the ist of connected resources
resources_list = rm.list_resources()
# stores the first resource listed
instr_visa_address = resources_list[0]
print('Resource instr_visa_address: ', instr_visa_address)

wfgen = rm.open_resource(instr_visa_address)
print('Instrument info:', wfgen.query('*IDN?'))

# configure a sine wave
function = 'FUNC SIN'
freq = 'FREQ +1E+05'
volt_high = 'VOLT:HIGH +2.0'
volt_low = 'VOLT:LOW 0'
phase = 'PHASE 0'

command_list = [function, freq, volt_high, volt_low, phase, 'OUTP ON']

for command in command_list:
	wfgen.write(command)

while command != 'close':
	command = input()
	if command != 'close':
		wfgen.write(command)

wfgen.write('OUTP OFF')

wfgen.close()
rm.close()

print('end of program.')