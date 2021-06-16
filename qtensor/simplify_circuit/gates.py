"""
Created on Tue Oct 27 14:54:39 2020

@author: jonat, danlkv

Gates that contain information about commutation.

"""
from numpy import pi, sqrt


'''

First create a dummy menengere of gates so we can test the program.
The important things are knowing the eigenbasis per qubit and which qubits are acted upon

'''
class ident:
    '''
    The identity gate. This is important to preserve indexing of the gates.
    '''
    def __init__(self,angle=0):
        self.index = tuple()
        self.eigenbasis = {}
        self.angle = 0
    def __repr__(self):
        return '>> Identity <<'

class zzphase:
    def __init__(self,index1,index2,angle):
        assert index1!=index2
        self.index = tuple(sorted([index1,index2])) # indexing is the same so sort left-to-right.
        self.eigenbasis = {index1:'Z',index2:'Z'}
        self.angle = angle
    def __repr__(self):
        return '>> ZZphase on ({:0.0f} {:0.0f})[ {:0.2f} ] <<'.format(self.index[0],self.index[1],self.angle)

class xphase:
    def __init__(self,index1,angle):
        self.index = (index1,)
        self.eigenbasis = {index1:'X'}
        self.angle = angle
    def __repr__(self):
        return '>> Xphase on ({:0.0f})[{:0.2f}] <<'.format(self.index[0],self.angle)

class yphase:
    def __init__(self,index1,angle):
        self.index = (index1,)
        self.eigenbasis = {index1:'Y'}
        self.angle = angle
    def __repr__(self):
        return '>> Yphase on ({:0.0f})[{:0.2f}] <<'.format(self.index[0],self.angle)

class zphase:
    def __init__(self,index1,angle):
        self.index = (index1,)
        self.eigenbasis = {index1:'Z'}
        self.angle = angle
    def __repr__(self):
        return '>> Zphase on ({:0.0f})[{:0.2f}] <<'.format(self.index[0],self.angle)

class x:
    def __init__(self,index1):
        self.index = (index1,)
        self.eigenbasis = {index1:'X'}
        self.angle = 0
    def __repr__(self):
        return '>> X on ({:0.0f}) <<'.format(self.index[0],self.angle)

class y:
    def __init__(self,index1):
        self.index = (index1,)
        self.eigenbasis = {index1:'Y'}
        self.angle = 0
    def __repr__(self):
        return '>> Y on ({:0.0f}) <<'.format(self.index[0],self.angle)

class z:
    def __init__(self,index1):
        self.index = (index1,)
        self.eigenbasis = {index1:'Z'}
        self.angle = 0
    def __repr__(self):
        return '>> Z on ({:0.0f}) <<'.format(self.index[0],self.angle)

class hadamard:
    def __init__(self,index1,angle=0):
        self.index = (index1,)
        self.eigenbasis = {index1:'X+Y'}
        self.angle = pi/2/sqrt(2)
    def __repr__(self):
        return '>> Hadamard on ({:0.0f}) <<'.format(self.index[0])

class cnot:
    def __init__(self,index1,index2,angle=0):
        self.index = (index1,index2)
        self.eigenbasis = {index1:'cnot_A',index2:'cnot_B'} # Basis is entangled; this is one way to describe it...
        self.angle = 0
    def __repr__(self):
        return '>> CNOT on ({:0.0f} : {:0.0f}) <<'.format(self.index[0],self.index[1])

class cz:
    def __init__(self,index1,index2,angle=0):
        self.index = (index1,index2)
        self.eigenbasis = {index1:'cz_A',index2:'Z'} 
        self.angle = 0
    def __repr__(self):
        return '>> CNOT on ({:0.0f} : {:0.0f}) <<'.format(self.index[0],self.index[1])

class toffoli:
    def __init__(self,index1,index2,index3,angle=0):
        self.index = tuple(sorted([index1,index2])) + (index3,)
        self.eigenbasis = {index1:'toffoli_A',index2:'toffoli_A',index3:'toffoli_B'} # Basis is entangled; this is one way to describe it...
        self.angle = 0
    def __repr__(self):
        return '>> Toffoli on ({:0.0f} {:0.0f} : {:0.0f}) <<'.format(self.index[0],self.index[1],self.index[2])

