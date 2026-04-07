# Set terminal and output
set terminal pngcairo size 1000, 400  # Adjust the size to accommodate two plots
set output 'plots/plot_240829_xu_froment_fig9.png'
set loadpath '/home/meck/diss-code'; load "fststyle.cfg"

# Define the layout for the plots
set multiplot layout 1,2 title "Temperature dependence of rate and adsorption parameters."

# First plot: log10(k1), log10(k2), log10(k3)
set xlabel "1000 / T (1/K)"
set ylabel "log_{10}(k)"
set key bottom left

set xrange [1.2:1.8]
set yrange [-7:3]

set datafile sep ','

plot '-' using (1000/$1):(log10($2)) title "log_{10}k_1" with points pointtype 7 pointsize 1 lc rgb "black", \
     '-' using (1000/$1):(log10($2)) title "log_{10}k_2" with points pointtype 9 pointsize 1 lc rgb "black", \
     '-' using (1000/$1):(log10($2)) title "log_{10}k_3" with points pointtype 11 pointsize 1 lc rgb "black", \
     filename using (1000/$1):2 notitle with lines lc "black", \
     filename using (1000/$1):3 notitle with lines lc "black", \
     filename using (1000/$1):4 notitle with lines lc "black"

# Data block for k1
573.0,8.664e-07
598.0,4.655e-06
623.0,3.050e-05
648.0,1.626e-04
673.0,7.132e-04
773.0,2.088e-01
798.0,5.254e-01
823.0,2.069e+00
e

# Data block for k2
573.0,3.962e+00
598.0,4.481e+00
623.0,5.089e+00
648.0,7.333e+00
673.0,9.282e+00
e

# Data block for k3
573.0,1.965e-07
598.0,6.627e-07
623.0,5.149e-06
648.0,2.541e-05
673.0,1.457e-04
773.0,-1.216e-02
798.0,-5.070e-03
823.0,4.452e-01
e

# Second plot: log10(KCO), log10(KH2), log10(KCH4), log10(KH2O)
set xlabel "1000 / T (1/K)"
set ylabel "log_{10}(K)"
set key bottom left

set xrange [1.2:1.8]
set yrange [-7:3]

plot '-' using (1000/$1):(log10($2)) title "log_{10}K_{CO}" with points pointtype 7 pointsize 1 lc rgb "black", \
     '-' using (1000/$1):(log10($2)) title "log_{10}K_{H2}" with points pointtype 9 pointsize 1 lc rgb "black", \
     '-' using (1000/$1):(log10($2)) title "log_{10}K_{CH4}" with points pointtype 11 pointsize 1 lc rgb "black", \
     '-' using (1000/$1):(log10($2)) title "log_{10}K_{H2O}" with points pointtype 13 pointsize 1 lc rgb "black", \
     filename using ( (1000/$1 >= 1.45) ? (1000/$1) : 1/0 ):5 notitle with lines lc "black", \
     filename using ( (1000/$1 >= 1.5) ? (1000/$1) : 1/0 ):6 notitle with lines lc "black", \
     filename using ( (1000/$1 <= 1.3) ? (1000/$1) : 1/0 ):7 notitle with lines lc "black", \
     filename using ( (1000/$1 <= 1.3) ? (1000/$1) : 1/0 ):8 notitle with lines lc "black"

# Data block for KCO
573.0,417.10
598.0,104.40
623.0,59.90
648.0,32.35
673.0,23.60
e

# Data block for KH2
573.0,0.23170
598.0,0.11550
623.0,0.14000
648.0,0.09946
673.0,-0.05641
e

# Data block for KCH4
773.0,0.3218
798.0,0.2174
823.0,0.4356
e

# Data block for KH2O
773.0,0.1300
798.0,0.1999
823.0,0.6412
e

unset multiplot
