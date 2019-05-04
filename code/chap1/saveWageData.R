## From ISLR package, save Wage data as csv file

data(Wage, package = 'ISLR')
write.csv(Wage, file = '../../data/Wage.csv')
