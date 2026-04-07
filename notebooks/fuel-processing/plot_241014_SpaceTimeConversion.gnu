set loadpath '/home/meck/diss-code'; load "fststyle.cfg"
set datafile sep ','

# set key bottom left
# set key invert
set xrange [0:0.5]

# notebook version
set terminal pngcairo enhanced font 'Verdana,10'
set output "plots/241014_SpaceTimeConversion.png"

set xlabel 'Weight-time W/F_{CH_4,0} in g(cat) hr / mol'
set ylabel 'Conversion X_{CH_4}'

# f(x) = a*x+b
# fit [.4:.5] [*:*] f(x) filename \
#     u 1:2 via a,b

plot for [j=1:11] filename using 1:(column(j+1)) with l title sprintf('T = %s K',columnhead(j+1)) ls (2*j-1) lc 'black'