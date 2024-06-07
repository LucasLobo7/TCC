#!python3
# -*- coding: utf-8 -*-
# *********************************************************
# This program illustrates a few commonly-used programming
# features of your Keysight oscilloscope.
# *********************************************************
# Import modules.
# ---------------------------------------------------------
import pyvisa
import struct
import sys
# Global variables.
# ---------------------------------------------------------
# input_channel = "CHANnel1"
input_channel = "CHANnel2"
setup_file_name = "setup.scp"
screen_image_file_name = "screen_image.png"
waveform_data_file_name = "waveform_data.csv"
# wfm_fmt = "BYTE"
wfm_fmt = "WORD"

debug = False

# =========================================================
# Initialize:
# =========================================================
def initialize():
# Get and display the device's *IDN? string.
    idn_string = do_query_string("*IDN?")
    print(f"Identification string: '{idn_string}'")
    # Clear status and load the default setup.
    do_command("*CLS")
    do_command("*RST")

# =========================================================
# Capture:
# =========================================================
def capture():

    # Use auto-scale to automatically set up oscilloscope.
    print("Autoscale.")
    do_command(":AUToscale")

    # Set trigger mode.
    do_command(":TRIGger:MODE EDGE")
    qresult = do_query_string(":TRIGger:MODE?")
    print(f"Trigger mode: {qresult}")

    # Set EDGE trigger parameters.
    do_command(f":TRIGger:EDGE:SOURce {input_channel}")
    qresult = do_query_string(":TRIGger:EDGE:SOURce?")
    print(f"Trigger edge source: {qresult}")
    do_command(":TRIGger:EDGE:LEVel 1.5")
    qresult = do_query_string(":TRIGger:EDGE:LEVel?")
    print(f"Trigger edge level: {qresult}")
    do_command(":TRIGger:EDGE:SLOPe POSitive")
    qresult = do_query_string(":TRIGger:EDGE:SLOPe?")
    print(f"Trigger edge slope: {qresult}")

    # Save oscilloscope setup.
    setup_bytes = do_query_ieee_block(":SYSTem:SETup?")
    f = open(setup_file_name, "wb")
    f.write(setup_bytes)
    f.close()
    print(f"Setup bytes saved: {len(setup_bytes)}")

    # Change oscilloscope settings with individual commands:
    # Set vertical scale and offset.
    do_command(f":{input_channel}:SCALe 0.05")
    qresult = do_query_string(f":{input_channel}:SCALe?")
    print(f"{input_channel} vertical scale: {qresult}")
    do_command(f":{input_channel}:OFFSet -1.5")
    qresult = do_query_string(f":{input_channel}:OFFSet?")
    print(f"{input_channel} offset: {qresult}")

    # Set horizontal scale and offset.
    do_command(":TIMebase:SCALe 0.0002")
    qresult = do_query_string(":TIMebase:SCALe?")
    print(f"Timebase scale: {qresult}")
    do_command(":TIMebase:POSition 0.0")
    qresult = do_query_string(":TIMebase:POSition?")
    print(f"Timebase position: {qresult}")

    # Set the acquisition type.
    do_command(":ACQuire:TYPE NORMal")
    qresult = do_query_string(":ACQuire:TYPE?")
    print(f"Acquire type: {qresult}")

    # Or, set up oscilloscope by loading a previously saved setup.
    setup_bytes = ""
    f = open(setup_file_name, "rb")
    setup_bytes = f.read()
    f.close()
    do_command_ieee_block(":SYSTem:SETup", setup_bytes)
    print(f"Setup bytes restored: {len(setup_bytes)}")
    
    # Capture an acquisition using :DIGitize.
    do_command(f":DIGitize {input_channel}")

# =========================================================
# Analyze:
# =========================================================
def analyze():
    # Make measurements.
    # --------------------------------------------------------
    do_command(f":MEASure:SOURce {input_channel}")
    qresult = do_query_string(":MEASure:SOURce?")
    print(f"Measure source: {qresult}")
    do_command(":MEASure:FREQuency")
    qresult = do_query_string(":MEASure:FREQuency?")
    print(f"Measured frequency on {input_channel}: {qresult}")
    do_command(":MEASure:VAMPlitude")
    qresult = do_query_string(":MEASure:VAMPlitude?")
    print(f"Measured vertical amplitude on {input_channel}: {qresult}")

    # Download the screen image.
    # --------------------------------------------------------
    do_command(":HARDcopy:INKSaver OFF")
    screen_bytes = do_query_ieee_block(":DISPlay:DATA? PNG, COLor")
    # Save display data values to file.
    f = open(screen_image_file_name, "wb")
    f.write(screen_bytes)
    f.close()
    print(f"Screen image written to {screen_image_file_name}.")

    # Download waveform data.
    # --------------------------------------------------------

    # Set the waveform source.
    do_command(f":WAVeform:SOURce {input_channel}")
    qresult = do_query_string(":WAVeform:SOURce?")
    print(f"Waveform source: {qresult}")

    # Set the waveform points mode.
    do_command(":WAVeform:POINts:MODE RAW")
    qresult = do_query_string(":WAVeform:POINts:MODE?")
    print(f"Waveform points mode: {qresult}")

    # Get the number of waveform points available.
    do_command(":WAVeform:POINts 10240")
    qresult = do_query_string(":WAVeform:POINts?")
    print(f"Waveform points available: {qresult}")

    # Choose the format of the data returned (BYTE or WORD):
    do_command(f":WAVeform:FORMat {wfm_fmt}")
    qresult = do_query_string(":WAVeform:FORMat?")
    print(f"Waveform format: {qresult}")

    # Specify the byte order in WORD data.
    if wfm_fmt == "WORD":
        do_command(":WAVeform:BYTeorder LSBF")
        qresult = do_query_string(":WAVeform:BYTeorder?")
        print(f"Waveform byte order for WORD data: {qresult}")

    # Display the waveform settings from preamble:
    wav_form_dict = {
    0 : "BYTE",
    1 : "WORD",
    4 : "ASCii",
    }
    acq_type_dict = {
    0 : "NORMal",
    1 : "PEAK",
    2 : "AVERage",
    3 : "HRESolution",
    }
    preamble_string = do_query_string(":WAVeform:PREamble?")
    (
    wav_form, acq_type, wfmpts, avgcnt, x_increment, x_origin,
    x_reference, y_increment, y_origin, y_reference
    ) = preamble_string.split(",")
    print(f"Waveform format: {wav_form_dict[int(wav_form)]}")
    print(f"Acquire type: {acq_type_dict[int(acq_type)]}")
    print(f"Waveform points desired: {wfmpts}")

    print(f"Waveform average count: {avgcnt}")
    print(f"Waveform X increment: {x_increment}")
    print(f"Waveform X origin: {x_origin}")
    print(f"Waveform X reference: {x_reference}") # Always 0.
    print(f"Waveform Y increment: {y_increment}")
    print(f"Waveform Y origin: {y_origin}")
    print(f"Waveform Y reference: {y_reference}")
    
    # Get numeric values for later calculations.
    x_increment = do_query_number(":WAVeform:XINCrement?")
    x_origin = do_query_number(":WAVeform:XORigin?")
    y_increment = do_query_number(":WAVeform:YINCrement?")
    y_origin = do_query_number(":WAVeform:YORigin?")
    y_reference = do_query_number(":WAVeform:YREFerence?")

    # Get the waveform data.
    data_bytes = do_query_ieee_block(":WAVeform:DATA?")
    data_bytes_length = len(data_bytes)
    print(f"Byte count: {data_bytes_length}")
    if wfm_fmt == "BYTE":
        block_points = data_bytes_length
    elif wfm_fmt == "WORD":
        block_points = data_bytes_length / 2

    # Unpack or split into list of data values.
    if wfm_fmt == "BYTE":
        values = struct.unpack("%dB" % block_points, data_bytes)
    elif wfm_fmt == "WORD":
        values = struct.unpack("%dH" % block_points, data_bytes)
    print(f"Number of data values: {len(values)}")

    # Save waveform data values to CSV file.
    f = open(waveform_data_file_name, "w")

    for i in range(0, len(values) - 1):
        time_val = x_origin + (i * x_increment)
        voltage = ((values[i] - y_reference) * y_increment) + y_origin
        f.write(f"{time_val:E}, {voltage:f}\n")

    # Close output file.
    f.close()
    print(f"Waveform format {wfm_fmt} data written to {waveform_data_file_name}.")

# =========================================================
# Send a command and check for errors:
# =========================================================
def do_command(command, hide_params=False):
    if hide_params:
        (header, data) = command.split(" ", 1)
    if debug:
        print(f"Cmd = '{header}'")
    else:
        if debug:
            print(f"Cmd = '{command}'")
    
    InfiniiVision.write(f"{command}")

    if hide_params:
        check_instrument_errors(header)
    else:
        check_instrument_errors(command)

# =========================================================
# Send a command and binary values and check for errors:
# =========================================================
def do_command_ieee_block(command, values):
    if debug:
        print(f"Cmb = '{command}'")
    InfiniiVision.write_binary_values(f"{command} ", values, datatype='B')
    check_instrument_errors(command)

# =========================================================
# Send a query, check for errors, return string:
# =========================================================
def do_query_string(query):
    if debug:
        print(f"Qys = '{query}'")
    result = InfiniiVision.query(f"{query}")
    check_instrument_errors(query)
    
    return result.strip()

# =========================================================
# Send a query, check for errors, return floating-point value:
# =========================================================
def do_query_number(query):
    if debug:
        print(f"Qyn = '{query}'")
    results = InfiniiVision.query(f"{query}")
    check_instrument_errors(query)
    return float(results)

# =========================================================
# Send a query, check for errors, return binary values:
# =========================================================
def do_query_ieee_block(query):
    if debug:
        print(f"Qyb = '{query}'")
    result = InfiniiVision.query_binary_values(f"{query}", datatype='s', container=bytes)
    check_instrument_errors(query)
    return result

# =========================================================
# Check for instrument errors:
# =========================================================

def check_instrument_errors(command):
    while True:
        error_string = InfiniiVision.query(":SYSTem:ERRor?")
        if error_string: # If there is an error string value.
            if error_string.find("+0,", 0, 3) == -1: # Not "No error".
                print(f"ERROR: {error_string}, command: '{command}'")
                print("Exited because of error.")
                sys.exit(1)
            else: # "No error"
                break
        
        else: # :SYSTem:ERRor? should always return string.
            print(f"ERROR: :SYSTem:ERRor? returned nothing, command: '{command}'")
            print("Exited because of error.")
            sys.exit(1)

# =========================================================
# Main program:
# =========================================================
            
rm = pyvisa.ResourceManager("C:\\Windows\\System32\\visa64.dll")
InfiniiVision = rm.open_resource("TCPIP0::k-mx3104g-10104.cos.is.keysight.com::inst0::INSTR")
InfiniiVision.timeout = 15000
InfiniiVision.clear()

# Initialize the oscilloscope, capture data, and analyze.
initialize()
capture()
analyze()
InfiniiVision.close()
print("End of program.")
sys.exit()