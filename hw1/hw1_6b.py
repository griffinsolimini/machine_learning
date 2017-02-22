import math

err = 0.6 * 0.5 * ((math.erf(-7.583 / math.sqrt(2))) - math.erf(-float("inf")))

print err

err1 = 0.4 * 0.5 * (math.erf( (0.583-2) / (2 * math.sqrt(2))) - math.erf( -10.583 / (2*math.sqrt(2)) ))

print err1

err2 = 0.6 * 0.5 * (math.erf(float("inf")) - (math.erf(1.583 / math.sqrt(2)))) 

print err2

print err + err1 + err2

