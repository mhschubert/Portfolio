load("../../data/preprocessed/feature_data.Rdata")
library(ggplot2)

##test how many recid violent switch the category

table(features_red$recid_violent, features_red$recid_violent_corr)


##test correlation between recid and time in prison

time <- features_red$custody_days_jail_viol+features_red$custody_days_prison_viol

m <- glm(features_red$recid_violent ~time , family = 'binomial')
summary(m)




##test correlation between COMPAS decile scores and time

res <- cor.test(as.numeric(features_red$`Risk of Violence_decile_score`), time, 
               method = "pearson")
res


##get quantiles for time in jail+prison

quantile(time)

##lower 50% are 0-43 days
lower <- ifelse(time <4,0,1)

t.test(as.numeric(features_red$`Risk of Violence_decile_score`)~lower, var.equal = TRUE, alternative = "two.sided")

tab <- table(features_red$recid_violent, lower)
dimnames(tab) <- list(recid=c('F', 'T'),
                      quantile =c('lower', 'upper'))
chisq.test(tab)


ggplot(features_red[], aes(x=as.numeric(`Risk of Violence_decile_score`), y= custody_days_jail_viol))+
  geom_point()+
  geom_smooth()+
  scale_y_continuous(limits = c(0,1000))
