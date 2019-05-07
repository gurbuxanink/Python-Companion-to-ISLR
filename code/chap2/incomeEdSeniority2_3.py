# Plot Income versus Education and Seniority
# Use third order polynomial fit as "true" function
# import sys
# sys.path.append('./code/chap2/')
import incEdSen3d

my_formula = 'Income ~ Education + I(Education**2) + I(Education**3) + Seniority + I(Seniority**2) + I(Seniority**3)'

incEdSen3d.plotIncomeEdSeniority(my_formula)

