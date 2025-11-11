# The current volume of a water reservoir (in cubic metres)
reservoir_volume = 4.445e8
# The amount of rainfall from a storm (in cubic metres)
rainfall = 5e6
# decrease the rainfall variable by 10% to account for runoff
rainfall = rainfall*0.9
# add the rainfall variable to the reservoir_volume variable
total=reservoir_volume+rainfall
# increase reservoir_volume by 5% to account for stormwater that flows
reservoir_volume1=reservoir_volume*1.05
# into the reservoir in the days following the storm
# decrease reservoir_volume by 5% to account for evaporation
reservoir_volume2=reservoir_volume1*0.95
# subtract 2.5e5 cubic metres from reservoir_volume to account for water
reservoir_volume3=reservoir_volume2-2.5e5
# that's piped to arid regions.
# print the new value of the reservoir_volume variable
print(reservoir_volume3)