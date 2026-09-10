* Unreduced circuit exported from full_mna.py, explicit electrode laws
.param VT=0.0258641866305
.func sp(x) {max(x,0)+ln(1+exp(-abs(x)))}
Rsource in jack 10000
Rinput jack 0 1000000
Rstop1 jack g1 68000
B_Xv1b_plate p1 k1 I={2*max((v(p1,k1)/600*sp(600*(0.01+v(g1,k1)/sqrt(300+v(p1,k1)^2)))),0)^1.4/1060}
C_Xv1b_gk g1 k1 2.2999999999999999e-12
C_Xv1b_gp g1 p1 2.3999999999999999e-12
C_Xv1b_pk p1 k1 9e-13
R_Xv1b_pk p1 k1 1000000000
R_Xv1b_grid g1 xv1b_junction 2001
D_Xv1b_grid xv1b_junction k1 M_D_Xv1b_grid
.model M_D_Xv1b_grid D(Is=1.0000000000000001e-09 N=1 Rs=0 Cjo=9.9999999999999994e-12 Vj=1 M=0.5 Fc=0.5 Tt=1.0000000000000001e-09)
Ra1 bpre1 p1 100000
Rk1 k1 0 2700
Ck1 k1 0 6.8000000000000005e-07
Cc1 p1 inter1 2.2000000000000002e-08
Rpeak1 inter1 gain_top 470000
Cpeak1 inter1 gain_top 4.7000000000000003e-10
Rgain_top gain_top g2 900000
Rgain_bottom g2 0 100000
Cbright gain_top g2 1.0000000000000001e-09
B_Xv1a_plate p2 k2 I={2*max((v(p2,k2)/600*sp(600*(0.01+v(g2,k2)/sqrt(300+v(p2,k2)^2)))),0)^1.4/1060}
C_Xv1a_gk g2 k2 2.2999999999999999e-12
C_Xv1a_gp g2 p2 2.3999999999999999e-12
C_Xv1a_pk p2 k2 9e-13
R_Xv1a_pk p2 k2 1000000000
R_Xv1a_grid g2 xv1a_junction 2001
D_Xv1a_grid xv1a_junction k2 M_D_Xv1a_grid
.model M_D_Xv1a_grid D(Is=1.0000000000000001e-09 N=1 Rs=0 Cjo=9.9999999999999994e-12 Vj=1 M=0.5 Fc=0.5 Tt=1.0000000000000001e-09)
Ra2 bpre1 p2 100000
Rk2 k2 0 10000
Cc2 p2 inter2 2.2000000000000002e-08
Rpeak2 inter2 g3 470000
Cpeak2 inter2 g3 4.7000000000000003e-10
Rgrid3 g3 0 470000
B_Xv2a_plate p3 k3 I={2*max((v(p3,k3)/600*sp(600*(0.01+v(g3,k3)/sqrt(300+v(p3,k3)^2)))),0)^1.4/1060}
C_Xv2a_gk g3 k3 2.2999999999999999e-12
C_Xv2a_gp g3 p3 2.3999999999999999e-12
C_Xv2a_pk p3 k3 9e-13
R_Xv2a_pk p3 k3 1000000000
R_Xv2a_grid g3 xv2a_junction 2001
D_Xv2a_grid xv2a_junction k3 M_D_Xv2a_grid
.model M_D_Xv2a_grid D(Is=1.0000000000000001e-09 N=1 Rs=0 Cjo=9.9999999999999994e-12 Vj=1 M=0.5 Fc=0.5 Tt=1.0000000000000001e-09)
Ra3 bpre2 p3 100000
Rk3 k3 0 820
B_Xv2b_plate bpre2 cf I={2*max((v(bpre2,cf)/600*sp(600*(0.01+v(p3,cf)/sqrt(300+v(bpre2,cf)^2)))),0)^1.4/1060}
C_Xv2b_gk p3 cf 2.2999999999999999e-12
C_Xv2b_gp p3 bpre2 2.3999999999999999e-12
C_Xv2b_pk bpre2 cf 9e-13
R_Xv2b_pk bpre2 cf 1000000000
R_Xv2b_grid p3 xv2b_junction 2001
D_Xv2b_grid xv2b_junction cf M_D_Xv2b_grid
.model M_D_Xv2b_grid D(Is=1.0000000000000001e-09 N=1 Rs=0 Cjo=9.9999999999999994e-12 Vj=1 M=0.5 Fc=0.5 Tt=1.0000000000000001e-09)
Rkcf cf 0 100000
Rslope cf ts 33000
Ctreble cf treble_top 4.7000000000000003e-10
Cbass ts bass_top 2.2000000000000002e-08
Cmid ts mid_wiper 2.2000000000000002e-08
Rtreble_top treble_top tone 125000
Rtreble_bottom tone bass_top 125000
Rbass bass_top mid_top 500000
Rmid_top mid_top mid_wiper 12500
Rmid_bottom mid_wiper 0 12500
Rmaster_top tone master_out 900000
Rmaster_bottom master_out 0 100000
Cpi master_out pi_g1 2.2000000000000002e-08
Rgpi1 pi_g1 pi_tail 1000000
Rgpi2 pi_g2 pi_tail 1000000
Cpi2 pi_g2 feedback 1.0000000000000001e-07
B_Xv3a_plate pi_p1 pi_k I={2*max((v(pi_p1,pi_k)/600*sp(600*(0.01+v(pi_g1,pi_k)/sqrt(300+v(pi_p1,pi_k)^2)))),0)^1.4/1060}
C_Xv3a_gk pi_g1 pi_k 2.2999999999999999e-12
C_Xv3a_gp pi_g1 pi_p1 2.3999999999999999e-12
C_Xv3a_pk pi_p1 pi_k 9e-13
R_Xv3a_pk pi_p1 pi_k 1000000000
R_Xv3a_grid pi_g1 xv3a_junction 2001
D_Xv3a_grid xv3a_junction pi_k M_D_Xv3a_grid
.model M_D_Xv3a_grid D(Is=1.0000000000000001e-09 N=1 Rs=0 Cjo=9.9999999999999994e-12 Vj=1 M=0.5 Fc=0.5 Tt=1.0000000000000001e-09)
B_Xv3b_plate pi_p2 pi_k I={2*max((v(pi_p2,pi_k)/600*sp(600*(0.01+v(pi_g2,pi_k)/sqrt(300+v(pi_p2,pi_k)^2)))),0)^1.4/1060}
C_Xv3b_gk pi_g2 pi_k 2.2999999999999999e-12
C_Xv3b_gp pi_g2 pi_p2 2.3999999999999999e-12
C_Xv3b_pk pi_p2 pi_k 9e-13
R_Xv3b_pk pi_p2 pi_k 1000000000
R_Xv3b_grid pi_g2 xv3b_junction 2001
D_Xv3b_grid xv3b_junction pi_k M_D_Xv3b_grid
.model M_D_Xv3b_grid D(Is=1.0000000000000001e-09 N=1 Rs=0 Cjo=9.9999999999999994e-12 Vj=1 M=0.5 Fc=0.5 Tt=1.0000000000000001e-09)
Rapi1 bpi pi_p1 82000
Rapi2 bpi pi_p2 100000
Cpi_hf pi_p1 pi_p2 4.6999999999999999e-11
Rpi_k pi_k pi_tail 470
Rtail pi_tail feedback 10000
Rfeedback_ground feedback 0 4700
Cpresence feedback pres 1.0000000000000001e-07
Rpresence pres 0 11000
Rnfb tap4 feedback 100000
Cdrive1 pi_p1 drive1 2.2000000000000002e-08
Cdrive2 pi_p2 drive2 2.2000000000000002e-08
Rbias1 drive1 bias 220000
Rbias2 drive2 bias 220000
Rg4 drive1 g4 5600
Rg5 drive1 g5 5600
Rg6 drive2 g6 5600
Rg7 drive2 g7 5600
Rs4 bscreen s4 1000
Rs5 bscreen s5 1000
Rs6 bscreen s6 1000
Rs7 bscreen s7 1000
B_Xv4_screen s4 0 I={max(v(s4,0)/11+v(g4,0),0)^1.35/4200}
B_Xv4_plate plate_a 0 I={2*max((max(v(s4,0),0)/60*sp(60*(1/11+v(g4,0)/max(v(s4,0),0.001)))),0)^1.35/650*atan(max(v(plate_a,0),0)/24)}
C_Xv4_gk g4 0 1.5e-11
C_Xv4_gp g4 plate_a 9.9999999999999998e-13
C_Xv4_pk plate_a 0 7.9999999999999998e-12
R_Xv4_pk plate_a 0 1000000000
R_Xv4_grid g4 xv4_junction 1001
D_Xv4_grid xv4_junction 0 M_D_Xv4_grid
.model M_D_Xv4_grid D(Is=1.0000000000000001e-09 N=1 Rs=0 Cjo=9.9999999999999994e-12 Vj=1 M=0.5 Fc=0.5 Tt=1.0000000000000001e-09)
B_Xv5_screen s5 0 I={max(v(s5,0)/11+v(g5,0),0)^1.35/4200}
B_Xv5_plate plate_a 0 I={2*max((max(v(s5,0),0)/60*sp(60*(1/11+v(g5,0)/max(v(s5,0),0.001)))),0)^1.35/650*atan(max(v(plate_a,0),0)/24)}
C_Xv5_gk g5 0 1.5e-11
C_Xv5_gp g5 plate_a 9.9999999999999998e-13
C_Xv5_pk plate_a 0 7.9999999999999998e-12
R_Xv5_pk plate_a 0 1000000000
R_Xv5_grid g5 xv5_junction 1001
D_Xv5_grid xv5_junction 0 M_D_Xv5_grid
.model M_D_Xv5_grid D(Is=1.0000000000000001e-09 N=1 Rs=0 Cjo=9.9999999999999994e-12 Vj=1 M=0.5 Fc=0.5 Tt=1.0000000000000001e-09)
B_Xv6_screen s6 0 I={max(v(s6,0)/11+v(g6,0),0)^1.35/4200}
B_Xv6_plate plate_b 0 I={2*max((max(v(s6,0),0)/60*sp(60*(1/11+v(g6,0)/max(v(s6,0),0.001)))),0)^1.35/650*atan(max(v(plate_b,0),0)/24)}
C_Xv6_gk g6 0 1.5e-11
C_Xv6_gp g6 plate_b 9.9999999999999998e-13
C_Xv6_pk plate_b 0 7.9999999999999998e-12
R_Xv6_pk plate_b 0 1000000000
R_Xv6_grid g6 xv6_junction 1001
D_Xv6_grid xv6_junction 0 M_D_Xv6_grid
.model M_D_Xv6_grid D(Is=1.0000000000000001e-09 N=1 Rs=0 Cjo=9.9999999999999994e-12 Vj=1 M=0.5 Fc=0.5 Tt=1.0000000000000001e-09)
B_Xv7_screen s7 0 I={max(v(s7,0)/11+v(g7,0),0)^1.35/4200}
B_Xv7_plate plate_b 0 I={2*max((max(v(s7,0),0)/60*sp(60*(1/11+v(g7,0)/max(v(s7,0),0.001)))),0)^1.35/650*atan(max(v(plate_b,0),0)/24)}
C_Xv7_gk g7 0 1.5e-11
C_Xv7_gp g7 plate_b 9.9999999999999998e-13
C_Xv7_pk plate_b 0 7.9999999999999998e-12
R_Xv7_pk plate_b 0 1000000000
R_Xv7_grid g7 xv7_junction 1001
D_Xv7_grid xv7_junction 0 M_D_Xv7_grid
.model M_D_Xv7_grid D(Is=1.0000000000000001e-09 N=1 Rs=0 Cjo=9.9999999999999994e-12 Vj=1 M=0.5 Fc=0.5 Tt=1.0000000000000001e-09)
Rpa plate_a oa 16.559999999999999
Rpb plate_b ob 15.359999999999999
Lpa oa bplus 2.2124999999999999
Lpb bplus ob 2.2124999999999999
Rload out 0 16
Cmain1 reservoir main_mid 9.9999999999999991e-05
Cmain2 main_mid 0 9.9999999999999991e-05
Lch bplus choke_out 3
Rch choke_out bscreen 100
Cscreen1 bscreen screen_mid 9.9999999999999991e-05
Cscreen2 screen_mid 0 9.9999999999999991e-05
Rscreenbalance1 bscreen screen_mid 56000
Rscreenbalance2 screen_mid 0 56000
Rdrop1 bscreen drop_mid 10000
Rdrop2 drop_mid bpi 10000
Cpi_supply bpi 0 9.9999999999999991e-05
Rdrop3 bpi bpre2 10000
Cpre2 bpre2 0 4.9999999999999996e-05
Rdrop4 bpre2 bpre1 10000
Cpre1 bpre1 0 4.9999999999999996e-05
Vin in 0 DC 0 SIN(0 0.001 1000 0 0 0)
Vmains line 0 DC 0 SIN(0 325.26911934581187 50 0 0 0)
Rmains_closed line primary 5
Lprimary primary 0 40
Lhv1 hv1i main_mid 23.156899810964084
Lhv2 main_mid hv2i 23.156899810964084
Rhv1 hv1i hv1 15
Rhv2 hv2i hv2 15
Lbiasw bias_ac_i 0 7.5614366729678641
Rbiasw bias_ac_i bias_ac 100
Lheater1 heater1i 0 0.0075028355387523615
Lheater2 0 heater2i 0.0075028355387523615
Rheater1 heater1i heater1 0.014999999999999999
Rheater2 heater2i heater2 0.014999999999999999
Rheater_load heater1 heater2 0.91304347826086951
Kpower Lprimary Lhv1 Lhv2 Lbiasw Lheater1 Lheater2 0.99990000000000001
R_Dbridge1 hv1 Dbridge1_a 0.050000000000000003
Dbridge1 Dbridge1_a reservoir M_Dbridge1
.model M_Dbridge1 D(Is=5.0000000000000001e-09 N=1 Rs=0 Cjo=1.9999999999999999e-11 Vj=1 M=0.5 Fc=0.5 Tt=1.9999999999999999e-06)
R_Dbridge2 hv2 Dbridge2_a 0.050000000000000003
Dbridge2 Dbridge2_a reservoir M_Dbridge2
.model M_Dbridge2 D(Is=5.0000000000000001e-09 N=1 Rs=0 Cjo=1.9999999999999999e-11 Vj=1 M=0.5 Fc=0.5 Tt=1.9999999999999999e-06)
R_Dbridge3 0 Dbridge3_a 0.050000000000000003
Dbridge3 Dbridge3_a hv1 M_Dbridge3
.model M_Dbridge3 D(Is=5.0000000000000001e-09 N=1 Rs=0 Cjo=1.9999999999999999e-11 Vj=1 M=0.5 Fc=0.5 Tt=1.9999999999999999e-06)
R_Dbridge4 0 Dbridge4_a 0.050000000000000003
Dbridge4 Dbridge4_a hv2 M_Dbridge4
.model M_Dbridge4 D(Is=5.0000000000000001e-09 N=1 Rs=0 Cjo=1.9999999999999999e-11 Vj=1 M=0.5 Fc=0.5 Tt=1.9999999999999999e-06)
Rht_closed reservoir bplus 0.01
Rbias_feed bias_ac bias_d 27000
Dbias bias_raw bias_d M_Dbias
.model M_Dbias D(Is=5.0000000000000001e-09 N=1 Rs=0 Cjo=1.9999999999999999e-11 Vj=1 M=0.5 Fc=0.5 Tt=1.9999999999999999e-06)
Cbias_raw bias_raw 0 1.0000000000000001e-05
Rbias_filter bias_raw bias 15000
Cbias bias 0 1.0000000000000001e-05
Rbias_bottom bias bias_pot 56000
Rbias_adjust bias_pot 0 11000
Lsec4 sec4i 0 0.020823529411764706
Rsec4 sec4i tap4 0.16
Lsec8 sec8i tap4 0.0035727528141087489
Rsec8 sec8i tap8 0.066274169979695222
Lsec16 sec16i tap8 0.0071455056282174918
Rsec16 sec16i out 0.093725830020304782
Koutput Lpa Lpb Lsec4 Lsec8 Lsec16 0.99950000000000006
