import pyvisa as pv
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio   
import pickle
import scipy.signal
from tqdm.notebook import tqdm, trange
from scipy.special import erfc
import scipy as sp
from optic.comm.modulation import modulateGray, demodulateGray, GrayMapping
from optic.dsp.core import firFilter, pulseShape, lowPassFIR, pnorm, upsample
from optic.comm.metrics import signal_power,fastBERcalc

from optic.plot import eyediagram
plt.rcParams["figure.figsize"] = (12,6)

def Gerar_Simbolos(M,nsimbolos,SPS,formatoPulso,nTaps,alpha):
    # Geração de simbolos
    bits = np.random.randint(0,2,int(nsimbolos*np.log2(M)))

    simbolos = modulateGray(bits, M, 'pam')
    simbolos = pnorm(simbolos)

    # Fomartação de pulso
    simbolosup = upsample(simbolos, SPS)
    pulso = pulseShape(formatoPulso, SPS, nTaps, alpha)
    pulso = pulso/max(abs(pulso))
    sinal = firFilter(pulso, simbolosup)
    sinal = sinal.real
    # if formatoPulso == 'rrc':
    #     np.savez('Dados Gerador De sinal/Python/{}PAM_SPS={}_{}_alpha={}.npz'.format(int(M),int(SPS),formatoPulso,alpha), simbolos=simbolos,sinal=sinal)
    # else:
    #     np.savez('Dados Gerador De sinal/Python/{}PAM_SPS={}_{}.npz'.format(int(M),int(SPS),formatoPulso), simbolos=simbolos,sinal=sinal)

    #npz = np.load('{}PAM_SPS={}_{}_aplha={}.npz'.format(int(M),int(SPS),formatoPulso,alpha))
    #print(npz['simbolos'])

    # Geração do arquivo do DAC
    sinal = sinal - np.min(sinal)
    sinal = sinal/np.max(sinal)*65534
    sinal = sinal - 32767
    sinal = (np.rint(sinal)).astype(int)
    return bits,sinal

def Onda_Dac(DAC,Porta,fs,V_High,V_Low,pontos,filtro):
    DAC.write(f':SOURce{Porta}:FUNCtion:SEQuence:STATe ON')
    DAC.write(f':SOURce{Porta}:FUNCtion:SEQuence:LIST:CLEar')
    tamanhopacote = 300
    simboloscortados = np.array_split(pontos,int(np.ceil(len(pontos)/tamanhopacote)))
    if len(simboloscortados) == 1:
        pontos = np.array2string(simboloscortados[0], separator=', ').translate({ord(j): None for j in '[]'}).replace('\n','')
        DAC.write(f':SOURce{Porta}:TRACe:DATA:DAC16 CODE,END, {pontos}')
        DAC.query('*OPC?')
    else:
        for i in range(len(simboloscortados)):
            pontos = np.array2string(simboloscortados[i], separator=', ').translate({ord(j): None for j in '[]'}).replace('\n','')
            if i == 0:
                DAC.write(f':SOURce{Porta}:TRACe:DATA:DAC16 CODE,HEADer, {pontos}')
                DAC.query('*OPC?')        
            elif i == len(simboloscortados) - 1:
                DAC.write(f':SOURce{Porta}:TRACe:DATA:DAC16 CODE,END, {pontos}')
                DAC.query('*OPC?')
            else:
                DAC.write(f':SOURce{Porta}:TRACe:DATA:DAC16 CODE,CONTinue, {pontos}')
                DAC.query('*OPC?')

    DAC.write(f':SOURce{Porta}:FUNCtion:SEQuence:LIST:SRATe {fs}')
    DAC.write(f':SOURce{Porta}:FUNCtion:SEQuence:LIST:FILTer {filtro}')
    DAC.write(f':SOURce{Porta}:FUNCtion:SEQuence:LIST:APPLy')

    DAC.write(f':SOURce{Porta}:VOLTage:HIGH {V_High}')
    DAC.write(f':SOURce{Porta}:VOLTage:LOW {V_Low}')
    DAC.write(f':OUTPut{Porta}:STATe ON')

def ConfigurarScope(scope,tempo,canais,vDivisao,impedancia,trigger,offset):
    scope.write('trigger:mode edge')
    scope.write(f'timebase:range {tempo}')
    scope.write(':CHANnel1:DISPlay 0')
    scope.write(':CHANnel2:DISPlay 0')
    scope.write(':CHANnel3:DISPlay 0')
    scope.write(':CHANnel4:DISPlay 0')
    for i in range(len(canais)):
        scope.write(f'trigger:level channel{canais[i]}, {trigger[i]}')
        scope.write(f'channel{canais[i]}:impedance {impedancia[i]}')
        scope.write(f':CHANnel{canais[i]}:OFFSet {offset[i]}')
        scope.write(f'CHANnel{canais[i]}:SCALe {vDivisao[i]}')
        scope.write(f':CHANnel{canais[i]}:DISPlay 1;*OPC?')
    

def AdquirirOnda(scope,canal):
    if canal == 'FFT':
        scope.write(':WAVeform:SOURce FFT')
    else:
        scope.write(f'waveform:source channel{canal}')
    scope.write('waveform:format byte')
    
    dados = scope.query_binary_values('waveform:data?', datatype='B')
    nDados = len(dados)

    tInicial = float(scope.query('waveform:xorigin?'))
    Δt = float(scope.query('waveform:xincrement?'))

    yIncial = float(scope.query('waveform:yorigin?'))
    Δy = float(scope.query('waveform:yincrement?'))
    yReferencia = float(scope.query('waveform:yreference?'))
    
    
    t = np.linspace(tInicial,tInicial + Δt*nDados,nDados,endpoint=0)
    y = (np.array(dados) - yReferencia)*Δy  + yIncial

    return t,y
def ConfigFFT(scope,canal,escaladB,fstart,fstop):
    scope.write(f':FFT:SOURce{canal}')
    scope.write(f':FFT:SCALe {escaladB}')
    scope.write(f':FFT:FREQuency:STARt {fstart}')
    scope.write(f':FFT:FREQuency:STOP {fstop}')
    scope.write(':FFT:DMODe AVERage')
    scope.write(':FFT:DISPlay 1')

def DemodularSinal(scope,scopeSinal,ScopeReferencia,AmplitudeReferencia,nSimbolos,SPS,simbolosTransmitidos,plot):
    tsinalizacao,ysinalizacao = AdquirirOnda(scope,ScopeReferencia)
    t,y = AdquirirOnda(scope,scopeSinal)
    if plot==True:
        plt.figure(1)
        plt.plot(tsinalizacao,ysinalizacao,label='Sinal de refencia')
        peak = sp.signal.find_peaks(ysinalizacao,0.9*AmplitudeReferencia,distance=len(ysinalizacao)//4)
        for i in range(len(peak[0])):
            plt.plot(tsinalizacao[peak[0][i]],ysinalizacao[peak[0][i]],'o',label='Inicio da transmissão')
        plt.plot(t,y,label='Sinal Recebido')
        plt.legend()
        plt.title('Sinais do osciloscopio')
        plt.ylabel('Amplitude (V)')
        plt.xlabel('Tempo (s)')
    
    ycortado = y[peak[0][0]:peak[0][1]]
    simbolosrecebidos = ycortado[0::len(ycortado)//(nSimbolos*SPS)]
    simbolosrecebidos = simbolosrecebidos[0:-1]
    if plot==True:
        plt.figure(2)
        simbolosTransmitidos = simbolosTransmitidos - np.min(simbolosTransmitidos)
        simbolosTransmitidos = (simbolosTransmitidos/np.max(simbolosTransmitidos))*3
        plt.plot(simbolosTransmitidos,'-o',label='Simbolos transmitidos')
        plt.plot(simbolosrecebidos/np.max(simbolosrecebidos)*3,'-o',label='Simbolos recebidos')
        plt.legend()
        plt.xlabel('Indice')
        plt.ylabel('Valor do Simbolo')
        plt.title('Simbolos Demodulados')
    return simbolosTransmitidos,simbolosrecebidos
