setwd("~/MATH3092 Project 30685729")
library(ggplot2)
library(lubridate)
library(EnvStats)
library(PerformanceAnalytics)

data = read.csv("Data.csv")
data$X = as.Date(data$X)
data$SP500 = as.numeric(gsub(",", "", data$SP500))
colnames(data)=c("Date","SP500","AAPL","AMZN", "GOOGL", "NFLX")
par(mfrow = c(1, 1))
summary(data)

sp_plot <- ggplot(data, aes(x=Date, y=SP500)) +
  geom_line( color="steelblue") + 
  xlab("Date") + ylab("Index Value") + ggtitle("S&P 500 historic monthly data") +
  theme(axis.text.x=element_text(angle=60, hjust=1)) +
  ylim(1000,5000) + scale_x_date(date_breaks = "3 month", date_labels =  "%b %Y") 
sp_plot

n <- nrow(data)+1
data$SP_mnt_return = ((data[2:n, 2] - data[1:(n-1), 2])/data[1:(n-1), 2])
data$AAPL_mnt_return = ((data[2:n, 3] - data[1:(n-1), 3])/data[1:(n-1), 3])
data$AMZN_mnt_return = ((data[2:n, 4] - data[1:(n-1), 4])/data[1:(n-1), 4])
data$GOOGL_mnt_return = ((data[2:n, 5] - data[1:(n-1), 5])/data[1:(n-1), 5])
data$NFLX_mnt_return = ((data[2:n, 6] - data[1:(n-1), 6])/data[1:(n-1), 6])

par(mfrow = c(1,2))
hist(data$SP_mnt_return, col = 'skyblue3', breaks = 20, main="Histogram of S&P Monthly returns", 
     xlab="Monthly returns of S&P 500", xlim = c(-0.15,0.15), ylim=c(0,25))
datahist <- data.frame(x=data$SP_mnt_return[1:60])
epdfPlot(data$SP_mnt_return, main="Empirical Probability Density Function", xlab="Monthly returns of S&P 500")

message("S&P 500 monthly returns: Mean = ",mean(data$SP_mnt_return, na.rm=1), ", Standard Deviation = ", sd(data$SP_mnt_return, na.rm=1))

stock_plot <- ggplot(data, aes(x=Date)) + 
  geom_line(aes(y = AAPL, color = "AAPL")) + 
  geom_line(aes(y = AMZN, color = "AMZN")) + 
  geom_line(aes(y = GOOGL, color = "GOOGL")) + 
  geom_line(aes(y = NFLX, color = "NFLX")) +
  xlab("Date") + ylab("Stock Value($)") + ggtitle("Stock Prices") +
  theme(axis.text.x=element_text(angle=60, hjust=1)) +
  scale_x_date(date_breaks = "6 month", date_labels =  "%b %Y")
stock_plot

getSymbols("AAPL",from="2012-01-01",to="2017-01-01")
getSymbols("AMZN",from="2012-01-01",to="2017-01-01")
getSymbols("GOOGL",from="2012-01-01",to="2017-01-01")
getSymbols("NFLX",from="2012-01-01",to="2017-01-01")

AAPL%>%chartSeries(TA='addBBands();addVo();addMACD()',subset='2012')
AMZN%>%chartSeries(TA='addBBands();addVo();addMACD()',subset='2012')
GOOGL%>%chartSeries(TA='addBBands();addVo();addMACD()',subset='2012')
NFLX%>%chartSeries(TA='addBBands();addVo();addMACD()',subset='2012')

mean <- data.frame(AAPL = mean(data$AAPL_mnt_return, na.rm=1), AMZN = mean(data$AMZN_mnt_return, na.rm=1), 
                   GOOGL = mean(data$GOOGL_mnt_return, na.rm=1), NFLX = mean(data$NFLX_mnt_return, na.rm=1))
message("AAPL monthly returns: Mean = ", mean(data$AAPL_mnt_return, na.rm=1), ", Standard Deviation = ", sd(data$AAPL_mnt_return, na.rm=1), 
        "Sharpe Ratio = ", (mean(data$AAPL_mnt_return, na.rm=1)-(0.005/12))/sd(data$AAPL_mnt_return, na.rm=1))
message("AMZN monthly returns: Mean = ",mean(data$AMZN_mnt_return, na.rm=1), ", Standard Deviation = ", sd(data$AMZN_mnt_return, na.rm=1), 
        "Sharpe Ratio = ", (mean(data$AMZN_mnt_return, na.rm=1)-(0.005/12))/sd(data$AMZN_mnt_return, na.rm=1)) 
message("GOOGL monthly returns: Mean = ",mean(data$GOOGL_mnt_return, na.rm=1), ", Standard Deviation = ", sd(data$GOOGL_mnt_return, na.rm=1), 
        "Sharpe Ratio = ", (mean(data$GOOGL_mnt_return, na.rm=1)-(0.005/12))/sd(data$GOOGL_mnt_return, na.rm=1))
message("NFLX monthly returns: Mean = ",mean(data$NFLX_mnt_return, na.rm=1), ",   Standard Deviation = ", sd(data$NFLX_mnt_return, na.rm=1), 
        "Sharpe Ratio = ", (mean(data$NFLX_mnt_return, na.rm=1)-(0.005/12))/sd(data$NFLX_mnt_return, na.rm=1))

stocks <- data.frame(AAPL = c(data[1:60, 8]), AMZN = c(data[1:60, 9]), GOOGL = c(data[1:60, 10]), NFLX = c(data[1:60, 11]))
betas <- data.frame(AAPL = cov(data[1:60, 7], data[1:60, 8])/var(data$SP_mnt_return, na.rm=1), 
                    AMZN = cov(data[1:60, 7], data[1:60, 9])/var(data$SP_mnt_return, na.rm=1),
                    GOOGL = cov(data[1:60, 7], data[1:60, 10])/var(data$SP_mnt_return, na.rm=1),
                    NFLX = cov(data[1:60, 7], data[1:60, 11])/var(data$SP_mnt_return, na.rm=1))
monthly_return_CAPM <- (0.005/12)+(mean(data$SP_mnt_return, na.rm=1)-(0.005/12))*betas
monthly_return_CAPM
difference <- (monthly_return_CAPM - mean)
round(cor(stocks),10)
round(cov(stocks),10)

var(data$AAPL_mnt_return, na.rm=1)


library(nloptr)
eval_f <- function(x)
{
  return ((x[1]*0.0719466254866474)^2 + (x[2]*0.0809196517969431)^2+(x[3]*0.0601019884976803)^2 +
            (x[4]*0.172085592402486)^2 + (2*x[1]*x[2]*0.0014105571873050500)+ (2*x[1]*x[3]*0.0012276893797342400)+ (2*x[1]*x[4]*-0.0011752915018600100)
          + (2*x[2]*x[3]*0.0025985030131040600)+ (2*x[2]*x[4]*0.0030024278011212800)+ (2*x[3]*x[4]*0.0025210840579824300))
}
eval_g_eq <- function(x)
{
  return ( x[1] + x[2] + x[3] + x[4] - 1 )
}
lb <- c(0,0,0,0)
ub <- c(1,1,1,1)
x0 <- c(0.25,0.25,0.25,0.25)
local_opts <- list( "algorithm" = "NLOPT_LD_MMA", "xtol_rel" = 1.0e-20 )
opts <- list( "algorithm"= "NLOPT_GN_ISRES",
              "xtol_rel"= 1.0e-20,
              "maxeval"= 160000,
              "local_opts" = local_opts,
              "print_level" = 0 )
res <- nloptr ( x0 = x0,
                eval_f = eval_f,
                lb = lb,
                ub = ub,
                eval_g_eq = eval_g_eq,
                opts = opts
)
print(res)
