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
from numpy.fft import fft, ifft
from optic.plot import eyediagram
plt.rcParams["figure.figsize"] = (12,6)

def Gerar_Simbolos(M,nsimbolos,SPS,formatoPulso,nTaps,alpha,bits=np.array([0])):
    ##############################################################
    # Função para gerar os pontos em PAM para serem enviados para o DAC
    # Os pontos ja saem escalonados entre -32767 e 32767 (short int, 2 bytes)

    # Parametros:
    # M (int): Formato de modulação
    # SPS (int): Amostras por simbolo
    # formatoPulso (string: rect, nrz, rrc): Formato do pulso
    # nTaps (int): Numero de taps do filtro rrc
    # alpha (float: entre 0 e 1): Rooloff do filtro rrc

    # Output:
    # bits (array numpy): Sequencia de bits gerados aleatoriamente
    # sinal: Sequencia de pontos formatada
    ##############################################################

    # Geração de simbolos
    if np.array_equal(bits,np.array([0])):
        bits = np.random.randint(0,2,int(nsimbolos*np.log2(M)))
    
    simbolos = modulateGray(bits, M, 'pam')
    simbolos = pnorm(simbolos)

    # Fomartação de pulso
    simbolosup = upsample(simbolos, SPS)
    pulso = pulseShape(formatoPulso, SPS, nTaps, alpha)
    pulso = pulso/max(abs(pulso))
    sinal = firFilter(pulso, simbolosup)
    sinal = sinal.real
    sinal = sinal - np.min(sinal)
    sinal = sinal/np.max(sinal)*65534
    sinal = sinal - 32767
    sinal = (np.rint(sinal)).astype(int)
    return bits,sinal


def Onda_Dac_Rigol(DAC,Porta,fs,V_High,V_Low,pontos,filtro):
    ##############################################################
    # Função para gerar a forma de onda arbitraria no DAC Rigol  dg922 Pro

    # Parametros:
    # DAC (Objeto do pyvisa relacionado ao DAC)
    # Porta (int 1 ou 2): Porta que será enviada a sequencia arbitraria
    # fs (float): Taxa de amostragem do DAC
    # V_High (float ou int): Amplitude maxima do sinal
    # V_Low  (float ou int): Amplitude minima do sinal
    # pontos (array numpy): Sequencia de pontos a ser carregada no DAC
    # filtro (string: Normal, Insert, Step): Formato de pulso usada no DAC
    ##############################################################

    DAC.write(f':SOURce{Porta}:FUNCtion:SEQuence:STATe ON')
    DAC.write(f':SOURce{Porta}:FUNCtion:SEQuence:LIST:CLEar')
    tamanhopacote = 2000
    simboloscortados = np.array_split(pontos,int(np.ceil(len(pontos)/tamanhopacote)))
    if len(simboloscortados) == 1:
        pontos = np.ndarray.tolist(simboloscortados[0])
        try:
            DAC.write_binary_values(f':SOURce{Porta}:TRACe:DATA:DAC16 BIN,END,', pontos,datatype='h')
            DAC.query('*OPC?')
        except:
                    print('tentando novamente')
                    DAC.query('*OPC?')
    else:
        for i in range(len(simboloscortados)):
            pontos = simboloscortados[i]
            if i == 0:
                try:
                    DAC.write_binary_values(f':SOURce{Porta}:TRACe:DATA:DAC16 BIN,HEADer,', pontos,datatype='h')
                    DAC.query('*OPC?')       
                except:
                    print(f'{i/len(simboloscortados)*100}%')
                    DAC.query('*OPC?') 
            elif i == len(simboloscortados) - 1:
                try:
                    DAC.write_binary_values(f':SOURce{Porta}:TRACe:DATA:DAC16 BIN,END,', pontos,datatype='h')
                    DAC.query('*OPC?')
                except:
                    print(f'{i/len(simboloscortados)*100}%')
                    DAC.query('*OPC?')
            else:
                try:
                    DAC.write_binary_values(f':SOURce{Porta}:TRACe:DATA:DAC16 BIN,CONTinue,', pontos,datatype='h')
                    DAC.query('*OPC?')
                except:
                    print(f'{i/len(simboloscortados)*100}%')
                    DAC.query('*OPC?')

    DAC.write(f':SOURce{Porta}:FUNCtion:SEQuence:LIST:SRATe {fs}')
    DAC.write(f':SOURce{Porta}:FUNCtion:SEQuence:LIST:FILTer {filtro}')
    DAC.write(f':SOURce{Porta}:FUNCtion:SEQuence:LIST:APPLy')

    DAC.write(f':SOURce{Porta}:VOLTage:HIGH {V_High}')
    DAC.write(f':SOURce{Porta}:VOLTage:LOW {V_Low}')
    DAC.write(f':OUTPut{Porta}:STATe ON')

def Onda_Dac_Keysight(DAC,Porta,fs,V_High,V_Low,pontos,Nome_Onda,filtro):
    ##############################################################
    # Função para gerar a forma de onda arbitraria no DAC Keysight Trueform 33600A  

    # Parametros:
    # DAC (Objeto do pyvisa relacionado ao DAC)
    # Porta (int 1 ou 2): Porta que será enviada a sequencia arbitraria
    # fs (float): Taxa de amostragem do DAC
    # V_High (float ou int): Amplitude maxima do sinal
    # V_Low  (float ou int): Amplitude minima do sinal
    # pontos (array numpy): Sequencia de pontos a ser carregada no DAC
    # Nome_Onda (string): Nome do arquivo de onda arbitrario
    # filtro (string: Normal, Insert, Step): Formato de pulso usada no DAC
    ##############################################################

    DAC.write(f'SOURce{Porta}:DATA:VOL:CLE')
    DAC.write('FORMat:BORDer SWAPped')
    DAC.write_binary_values('SOURCE{}:DATA:ARB:DAC {},'.format(Porta,Nome_Onda),pontos,datatype='h')
    print(DAC.query('SYSTEM:ERROR?'))  
    DAC.query('*OPC?')  
    DAC.write('SOURCE{}:FUNC ARB'.format(Porta))
    DAC.write('SOURCE{}:FUNC:ARB {}'.format(Porta,Nome_Onda)) 
    DAC.write('SOURCE{}:FUNC:ARB:SRAT {}'.format(Porta,fs))
    DAC.write(f'SOURCE{Porta}:FUNCtion:ARBitrary:FILTer {filtro}')
    DAC.write(f'SOURCE2:VOLT {(V_High-V_Low)}')
    DAC.write(f'SOURCE2:VOLT:OFFS {(V_High+V_Low)/2}')
    DAC.write('OUTP{} ON'.format(Porta))
    DAC.write('DISPLAY:FOCUS CH{}'.format(Porta))


def ConfigurarScope(scope,tempo,canais,vDivisao,impedancia,triggerChannel,triggerAmp,offset):
    ##############################################################
    # Função para configurar os canais e a visualização no Osciloscopio Keysight InfiniiVision DSOX3014T

    # Parametros:
    # scope (Objeto do pyvisa relacionado ao osciloscopio)
    # tempo (float): Intervalo de tempo na janela do osciloscopio
    # canais (int ou lista): Numero do/dos canais a serem configurados e ativados
    # vDvisao (int ou lista): Volts por divisão de cada canal
    # impedancia (string: FIFTy ou ONEMeg): Impedancia de cada canal
    # triggerChannel (int): Canal que será usado para trigger
    # triggerAmp (int): Amplitude do trigger
    # offset (int ou lista): Offset de cada canal
    ##############################################################

    scope.write('trigger:mode edge')
    scope.write(f'timebase:range {tempo}')
    scope.write(':CHANnel1:DISPlay 0')
    scope.write(':CHANnel2:DISPlay 0')
    scope.write(':CHANnel3:DISPlay 0')
    scope.write(':CHANnel4:DISPlay 0')
    for i in range(len(canais)):
        scope.write(f'channel{canais[i]}:impedance {impedancia[i]}')
        scope.write(f':CHANnel{canais[i]}:OFFSet {offset[i]}')
        scope.write(f'CHANnel{canais[i]}:SCALe {vDivisao[i]}')
        scope.write(f':CHANnel{canais[i]}:DISPlay 1;*OPC?')
    scope.write(f'trigger:level channel{triggerChannel}, {triggerAmp}')
    
def ConfigFFT(scope,canal,escaladB,fstart,fstop):
    ##############################################################
    # Função para ativar o modo FFT de um canal no Osciloscopio Keysight InfiniiVision DSOX3014T

    # Parametros:
    # scope (Objeto do pyvisa relacionado ao osciloscopio)
    # canal (int): Numero do canal que fará a fft
    # escaladB (float): Escala vertical da FFT
    # fstart (float): Frequencia inicial da FFT
    # fstop (float): Frequencia final da FFT
    ##############################################################

    scope.write(f':FFT:SOURce{canal}')
    scope.write(f':FFT:SCALe {escaladB}')
    scope.write(f':FFT:FREQuency:STARt {fstart}')
    scope.write(f':FFT:FREQuency:STOP {fstop}')
    scope.write(':FFT:DMODe AVERage')
    scope.write(':FFT:DISPlay 1')

def AdquirirOnda(scope,canal):
    ##############################################################
    # Função para retornar a forma de onda de um canal 

    # Parametros:
    # scope (Objeto do pyvisa relacionado ao osciloscopio)
    # canal (int ou string: FFT): Canal para obter a forma de onda

    # Outputs:
    # t (array numpy): array representando o tempo dos pontos do sinal ou frequencia caso escolha FFT
    # y (array numpy): array representando a amplitude dos pontos do sinal
    ##############################################################

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

def periodic_corr(x, y):
    ##############################################################
    # Função para calcular a correlação periodica entre dois arrays usando a FFT
    # Usada para calcular o atraso do sinal recebido.
    # Parametros:
    # x,y (array numpy): Arrays para calcular a correlação
    # Output:
    # Array da correlação dos dois sinais, para calcular o indice do atraso, bastar usar np.argmax()
    ##############################################################
    
    return (ifft(fft(x) * fft(y).conj()).real)

def DemodularSinal(scope,scopeSinal,ScopeReferencia,AmplitudeReferencia,nSimbolos,SPS,simbolosTransmitidos,plot):
    # Função Antiga para demodular o sinal do Osciloscopio
    # NÃO USADA MAIS
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

def LMS(x,d,L,μ,Niterações):
    x = np.append(np.zeros(L-1),x)
    W = np.zeros(L)
    erro = np.zeros(Niterações)

    for i in range(Niterações):
        xcortado = np.flip(x[i:L+i])
        erro[i] = d[i] - np.sum(W*xcortado)
        W = W + μ*erro[i]*xcortado
    return W,erro