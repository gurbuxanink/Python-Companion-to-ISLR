## From ISLR package, save Wage data as csv file

data(Wage, package = 'ISLR')
write.csv(Wage, file = 'data/Wage.csv')

## Save Smarket data as csv file
data(Smarket, package = 'ISLR')
write.csv(Smarket, file = 'data/Smarket.csv')

## Save NCI60 data, which is list with two components
## Each component is saved as a separate file
data(NCI60, package = 'ISLR')
write.csv(NCI60[['data']], file = 'data/NCI60data.csv') #this is a matrix
write.csv(NCI60[['labs']], file = 'data/NCI60labs.csv')
