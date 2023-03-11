from juliacall import Main as jl
import numpy as np

jl.seval("using Bora")
jl.seval("using AbstractEmulator")
jl.seval("using SimpleChains")
jl.seval("using BSON")
jl.seval("using Static")

__bora_compute_Xil = jl.seval('Bora.get_ξℓ')
__bora_compute_Xils = jl.seval('Bora.get_ξℓs')
__bora_compute_bb = jl.seval('Bora.get_broadband')
__load_emu_jl = jl.seval('BSON.load')

def compute_Xils(cosmo, emu):
    Pl = __bora_compute_Xils(jl.collect(cosmo), emu)
    return np.array(Pl)

def compute_Xils_vec(cosmo, emu):
    Pl = __bora_compute_Xils(jl.collect(np.transpose(cosmo)), emu)
    return np.array(Pl)

def compute_broadband(r, bbpar):
    return np.array(__bora_compute_bb(r, jl.collect(bbpar)))

def compute_broadband_vec(r, bbpar):
    return np.array(__bora_compute_bb(r, jl.collect(np.transpose(bbpar))))

def compute_Xil(cosmo, emu):
    Xil = __bora_compute_Xil(jl.collect(cosmo), emu)
    return np.array(Xil)

def compute_Xils_broadband(cosmo, bbpar, emu):
    return compute_Xils(cosmo, emu) + compute_broadband(emu.rgrid, bbpar)

def compute_Xils_broadband_vec(cosmo, bbpar, emu):
    return compute_Xils_vec(cosmo, emu) + compute_broadband_vec(emu.rgrid, bbpar)

def load_emu(path):
    loaded = __load_emu_jl(path)
    emu = loaded["ξℓs"]
    return emu

def get_rgrid(emu):
    return np.array(emu.rgrid)
