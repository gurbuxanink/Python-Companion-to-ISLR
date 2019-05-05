## From ISLR package, save Wage data as csv file

data(Wage, package = 'ISLR')
write.csv(Wage, file = 'data/Wage.csv')

## Save Smarket data as csv file
data(Smarket, package = 'ISLR')
write.csv(Smarket, file = 'data/Smarket.csv')
