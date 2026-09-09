* Marshall drawing pair 24-4-81: MASTER VOLUME + 100W EL34 branches.
* Complete signal path; provisional supply source / linear replacement OT.
* Local descriptive element names: factory sheets have no R/C designators.
.include tubes.lib
.param GAIN=.5 BASS=.5 MID=.5 TREBLE=.5 MASTER=.5 PRESENCE=.5
.param NFB=1 RSOURCE=10k VRAW=490 RPSU=120 VBIAS=-42
.param LPRI=8.85 RAA=1700 RLOAD=16 KOT=.9995
.param LCHOKE=3 RCHOKE=100
.func pot(x) {max(1m,min(.999,x))}

* High input; low input switching contacts closed, no plug in low jack.
Rsource in jack {RSOURCE}
Rinput jack 0 1Meg
Rstop1 jack g1 68k
Xv1b p1 g1 k1 ECC83_K
Ra1 bpre1 p1 100k
Rk1 k1 0 2.7k
Ck1 k1 0 .68u
Cc1 p1 inter1 22n
Rpeak1 inter1 gain_top 470k
Cpeak1 inter1 gain_top 470p
Rgain_top gain_top g2 {1Meg*(1-pot(GAIN))}
Rgain_bottom g2 0 {1Meg*pot(GAIN)}
Cbright gain_top g2 1n

* Cold-biased second stage, unbypassed 10k cathode.
Xv1a p2 g2 k2 ECC83_K
Ra2 bpre1 p2 100k
Rk2 k2 0 10k
Cc2 p2 inter2 22n
Rpeak2 inter2 g3 470k
Cpeak2 inter2 g3 470p
Rgrid3 g3 0 470k
Xv2a p3 g3 k3 ECC83_K
Ra3 bpre2 p3 100k
Rk3 k3 0 820

* Direct coupled cathode follower and loaded tone stack.
Xv2b bpre2 p3 cf ECC83_K
Rkcf cf 0 100k
Rslope cf ts 33k
Ctreble cf treble_top 470p
Cbass ts bass_top 22n
Cmid ts mid_wiper 22n
Rtreble_top treble_top tone {250k*(1-pot(TREBLE))}
Rtreble_bottom tone bass_top {250k*pot(TREBLE)}
Rbass bass_top mid_top {1Meg*pot(BASS)}
Rmid_top mid_top mid_wiper {25k*(1-pot(MID))}
Rmid_bottom mid_wiper 0 {25k*pot(MID)}
Rmaster_top tone master_out {1Meg*(1-pot(MASTER))}
Rmaster_bottom master_out 0 {1Meg*pot(MASTER)}

* Long-tailed-pair phase inverter, presence and output feedback.
Cpi master_out pi_g1 22n
Rgpi1 pi_g1 pi_tail 1Meg
Rgpi2 pi_g2 pi_tail 1Meg
Cpi2 pi_g2 feedback 100n
Xv3a pi_p1 pi_g1 pi_k ECC83_K
Xv3b pi_p2 pi_g2 pi_k ECC83_K
Rapi1 bpi pi_p1 82k
Rapi2 bpi pi_p2 100k
Cpi_hf pi_p1 pi_p2 47p
Rpi_k pi_k pi_tail 470
Rtail pi_tail feedback 10k
Rfeedback_ground feedback 0 4.7k
Cpresence feedback pres 100n
Rpresence pres 0 {22k*(1-pot(PRESENCE))}
Rnfb tap4 feedback {100k/max(NFB,1e-9)}

* Four individual EL34s; two per side, each with grid/screen resistor.
Cdrive1 pi_p1 drive1 22n
Cdrive2 pi_p2 drive2 22n
Rbias1 drive1 bias 220k
Rbias2 drive2 bias 220k
Rg4 drive1 g4 5.6k
Rg5 drive1 g5 5.6k
Rg6 drive2 g6 5.6k
Rg7 drive2 g7 5.6k
Rs4 bscreen s4 1k
Rs5 bscreen s5 1k
Rs6 bscreen s6 1k
Rs7 bscreen s7 1k
Xv4 plate_a g4 0 s4 EL34_K
Xv5 plate_a g5 0 s5 EL34_K
Xv6 plate_b g6 0 s6 EL34_K
Xv7 plate_b g7 0 s7 EL34_K

* Linear OT proxy: Hammond 1750U data, NOT measured original Dagnall.
* Full-primary L is 4 times half-primary L for ideal coupling.
Rpa plate_a oa 16.56
Rpb plate_b ob 15.36
Lpa oa bplus {LPRI/4}
Lpb bplus ob {LPRI/4}
Ls sec 0 {LPRI*16/RAA}
Kmain Lpa Lpb Ls {KOT}
Rsec sec out .320
Rload out 0 {RLOAD}
* Unloaded ideal 4-ohm feedback tap, half of 16-ohm winding voltage.
Etap tap4 0 sec 0 .5

* Reservoir / screen / preamp storage and loading are solved together.
* Rectifier+mains equivalent is preliminary DC Thevenin, no mains ripple.
Vraw raw 0 {VRAW}
Rps raw bplus {RPSU}
Cmain1 bplus main_mid 100u
Cmain2 main_mid 0 100u
Rmainbalance1 bplus main_mid 1G
Rmainbalance2 main_mid 0 1G
Lch bplus choke_out {LCHOKE}
Rch choke_out bscreen {RCHOKE}
Cscreen1 bscreen screen_mid 100u
Cscreen2 screen_mid 0 100u
Rscreenbalance1 bscreen screen_mid 56k
Rscreenbalance2 screen_mid 0 56k
Rdrop1 bscreen drop_mid 10k
Rdrop2 drop_mid bpi 10k
Cpi_supply bpi 0 100u
Rdrop3 bpi bpre2 10k
Cpre2 bpre2 0 50u
Rdrop4 bpre2 bpre1 10k
Cpre1 bpre1 0 50u
* Bias rectifier/network not yet transcribed: fixed source baseline.
Vbias bias 0 {VBIAS}
