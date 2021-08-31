#@Author: Marcel H. Schubert
#script evaluates the simulation reevaluation and plots results

require(ggplot2)
require(plyr)

df <- readRDS(file = '../Data/simulation/reevalution_frame.rds')

df <- df[, 1:60]
df[is.na(df$gamma_distance), 'gamma_distance'] <- 0
dl <- list()
#gamma = 0.1 + hierarchical + all cvis
s1 <- 'gamma_0.1_hierarchical_Sil-D-COP-DB-DBstar-CH-SF'

hg1 <- df[df$pick_success == s1,]
dl[[1]] <- hg1
names(dl)[1] <- s1 
#gamma = 0.1 + hierarchical + all cvis
s2 <- 'gamma_0.01_hierarchical_Sil-D-COP-DB-DBstar-CH-SF'

hg01 <- df[df$pick_success == s2,]
dl[[2]] <- hg01
names(dl)[2] <- s2 

#gamma = 0.001 + hierarchical + all cvis
s3 <- 'gamma_0.001_hierarchical_Sil-D-COP-DB-DBstar-CH-SF'

hg001 <- df[df$pick_success == s3,]
dl[[3]] <- hg001
names(dl)[3] <- s3 


hg <- df[grepl(paste(s1, s2, s3, sep='|'), df$pick_success), ]


print('hierarchical, all gammas')
round(prop.table(table(hg$gamma_distance, hg$k, dnn = c('gamma', '#k')), margin = 1), 2)
round(prop.table(table(hg$gamma_distance, hg$distance, dnn = c('gamma', 'distance'))),2)

print('hierarchical, gamma = 0.1')
round(prop.table(table(hg1[hg1$gamma_distance == 0.1,]$method, hg1[hg1$gamma_distance == 0.1,]$k,
                       dnn = c('method', '#k')), margin = 1), 2)
cdat1 <- ddply(hg1[hg1$gamma_distance == 0.1,], "method", summarise, k.mean=mean(k))
cdat1$gamma_distance = 0.1

print('hierarchical, gamma = 0.01')
round(prop.table(table(hg01[hg01$gamma_distance == 0.01,]$method, hg01[hg01$gamma_distance == 0.01,]$k,
                       dnn = c('method', '#k')), margin = 1), 2)
cdat01 <- ddply(hg01[hg01$gamma_distance == 0.01,], "method", summarise, k.mean=mean(k))
cdat01$gamma_distance = 0.01

print('hierarchical, gamma = 0.001')
round(prop.table(table(hg001[hg001$gamma_distance == 0.001,]$method, hg01[hg001$gamma_distance == 0.001,]$k,
                       dnn = c('method', '#k')), margin = 1), 2)
cdat001 <- ddply(hg001[hg001$gamma_distance == 0.001,], "method", summarise, k.mean=mean(k))
cdat001$gamma_distance = 0.001

cdat0 <- ddply(hg[hg$gamma_distance == 0,], "method", summarise, k.mean=mean(k))
cdat0$gamma_distance<- 0

cdat <- rbind(cdat0, cdat1, cdat01, cdat001)

df[is.na(df$gamma_distance), 'gamma_distance'] <- 0
hs <- df[df$pick_success =="gamma_all_hierarchical_Sil-D-COP-DB-DBstar-CH-SF",]

cdat1 <- ddply(hs[hs$gamma_distance == 0.1,], "method", summarise, k.mean=mean(k))
cdat1$gamma_distance = 0.1

cdat01 <- ddply(hs[hs$gamma_distance == 0.01,], "method", summarise, k.mean=mean(k))
cdat01$gamma_distance = 0.01

cdat0 <- ddply(hs[hs$gamma_distance == 0,], "method", summarise, k.mean=mean(k))
cdat0$gamma_distance <- 0

cdat <- rbind(cdat0, cdat1, cdat01)

gamma.labs <- c('dtw_basic', 'sdtw|gamma=0.001', 'sdtw|gamma=0.01','sdtw|gamma=0.1')
names(gamma.labs) <- as.character(c(0, 0.001, 0.01, 0.1))

p <- ggplot(hs, aes(x=k, color=method))+
  geom_density() +
  geom_vline(data=cdat, aes(xintercept=k.mean,  colour=method),
             linetype="dashed", size=1)+
  facet_grid(rows=vars(method), cols = vars(gamma_distance),
             labeller = labeller(gamma_distance = gamma.labs))+
  ggtitle('Distribution over best picks (given only hierarchical was used)')

p

ggsave(filename = '../figures/grid_search/reevaluation_gamma_method_hierarchical.pdf', height = 8, width = 10)







 s <- df[df$pick_success =="gamma_all_partitional_Sil-D-COP-DB-DBstar-CH-SF",]
s$gamma_distance[is.na(s$gamma_distance)] <- 0

round(prop.table(table(s$gamma_distance, s$k, dnn = c('gamma', '#k')), margin = 1), 2)

ss <- s[s$gamma_distance == 0.001,]
round(prop.table(table(s$gamma_distance, s$k, dnn = c('gamma', '#k')), margin = 2), 2)
round(prop.table(table(s$norm_distance, s$k, dnn = c('norm dist', '#k')), margin = 2), 2)
round(prop.table(table(s$norm_centroid, s$k, dnn = c('norm centr', '#k')), margin = 2), 2)


round(prop.table(table(ss$norm_distance, ss$k, dnn = c('norm dist', '#k')), margin = 2), 2)
round(prop.table(table(ss$norm_centroid, ss$k, dnn = c('norm centr', '#k')), margin = 2), 2)

p <- ggplot(s, aes(x=k))+
  geom_density() +
#  geom_vline(data=cdat, aes(xintercept=k.mean,  colour=method),
#             linetype="dashed", size=1)+
  facet_grid(cols = vars(gamma_distance),
             labeller = labeller(gamma_distance = gamma.labs))+
  ggtitle('Distribution over best picks (given only parttional was used)')

p

ggsave(filename = '../figures/grid_search/reevaluation_gamma_method_parttional.pdf', height = 8, width = 10)

