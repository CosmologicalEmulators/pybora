from juliacall import Main as jl
import numpy as np

jl.seval("using Bora")
jl.seval("using AbstractEmulator")
jl.seval("using SimpleChains")
jl.seval("using BSON")
jl.seval("using Static")

bora_compute_Xil = jl.seval('Bora.get_ξℓ')
bora_compute_Xils = jl.seval('Bora.get_ξℓs')
load_emu_jl = jl.seval('BSON.load')

def compute_Xils(*args):
    my_list = [elem for elem in args]
    if len(my_list) == 3:
        for i in range(len(args)-1):
            my_list[i] = jl.collect(my_list[i])
    else:
         for i in range(len(args)-2):
            my_list[i] = jl.collect(my_list[i])
    Pl = bora_compute_Xils(*my_list)
    return np.array(Pl)

def compute_Xil(cosmo, emu):
    Xil = bora_compute_Xil(jl.collect(cosmo), emu)
    return np.array(Xil)

def load_emu(path):
    loaded = load_emu_jl(path)
    emu = loaded["ξℓs"]
    return emu

def get_rgrid(emu):
    return np.array(emu.rgrid)
