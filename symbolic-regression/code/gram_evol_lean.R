#@Author: Marcel H. Schubert
#Script does the following:
#1. Cluster different public-goods game data using time-series clustering
#2. Use symbolic regression to find a function describing the indivudal clusters best

library(gramEvol)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(tidyr)
library(stringr)
library(pryr)
require(gridExtra) 
library(RColorBrewer)

df_exp <- read.csv("Data/exp_cmplt.csv")
colnames(df_exp)[which(names(df_exp) == "contr_nrmd")] <- "contribution"
colnames(df_exp)[which(names(df_exp) == "contr_oth_nrmd_lg")] <- "experience"

load("comparison_k7.Rdat") #TODO: load the right dataset!
source('utils/calc_rank.R') #calculate the rankvote to extract the correct specification

# extract time series from the chosen specification
clust_data<-comparison[["objects.partitional"]][[exp_id]]@datalist 
clust_data <- lapply(seq_along(clust_data), # make one DF out of list of matrices
                     function(i) {
                       data.frame(uid=i,
                                  contribution=clust_data[[i]][,"contribution"],
                                  experience=clust_data[[i]][,"experience"],
                                  period=2:(dim(clust_data[[i]])[1]+1))
                     })


centr <- comparison[["objects.partitional"]][[exp_id]]@centroids
centr <- lapply(seq_along(centr), # make one DF out of list of matrices
                     function(i) {
                       data.frame(cluster=i,
                                  contribution=centr[[i]][,"contribution"],
                                  experience=centr[[i]][,"experience"],
                                  period=2:(dim(centr[[i]])[1]+1))
                     })

clust_data<-do.call(rbind, clust_data)
centr<-do.call(rbind, centr)
centr$maxPeriod <- 7
# add cluster information
clust_clust<-as.data.frame(comparison[["objects.partitional"]][[exp_id]]@cluster)
clust_clust$uid<-1:dim(clust_clust)[1]
clust_data<-merge(clust_data, clust_clust, by = "uid")  # add cluster information
colnames(clust_data)[5]<-"cluster"


load("comparison_k10.Rdat") #TODO: load the right dataset!
source('utils/calc_rank.R') #calculate the rankvote to extract the correct specification

# extract time series from the chosen specification
tmp<-comparison[["objects.partitional"]][[exp_id]]@datalist 
tmp <- lapply(seq_along(tmp), # make one DF out of list of matrices
                     function(i) {
                       data.frame(uid=i,
                                  contribution=tmp[[i]][,"contribution"],
                                  experience=tmp[[i]][,"experience"],
                                  period=2:(dim(tmp[[i]])[1]+1))
                     })
tmp<-do.call(rbind, tmp)
tmpc <- comparison[["objects.partitional"]][[exp_id]]@centroids

tmpc <- lapply(seq_along(tmpc), # make one DF out of list of matrices
                function(i) {
                  data.frame(cluster=i,
                             contribution=tmpc[[i]][,"contribution"],
                             experience=tmpc[[i]][,"experience"],
                             period=2:(dim(tmpc[[i]])[1]+1))
                })


tmpc<-do.call(rbind, tmpc)
tmpc$maxPeriod <- 10

# add cluster information
clust_clust<-as.data.frame(comparison[["objects.partitional"]][[exp_id]]@cluster)
clust_clust$uid<-1:dim(clust_clust)[1]
tmp<-merge(tmp, clust_clust, by = "uid")  # add cluster information
colnames(tmp)[5]<-"cluster"

clust_data$maxPeriod = 7
tmp$maxPeriod = 10
clust_data <- rbind(clust_data, tmp)
centr <- rbind(centr, tmpc)


load("comparison_k20.Rdat") #TODO: load the right dataset!
source('utils/calc_rank.R') #calculate the rankvote to extract the correct specification

# extract time series from the chosen specification
tmp<-comparison[["objects.partitional"]][[exp_id]]@datalist 
tmp <- lapply(seq_along(tmp), # make one DF out of list of matrices
              function(i) {
                data.frame(uid=i,
                           contribution=tmp[[i]][,"contribution"],
                           experience=tmp[[i]][,"experience"],
                           period=2:(dim(tmp[[i]])[1]+1))
              })
tmp<-do.call(rbind, tmp)
tmpc <- comparison[["objects.partitional"]][[exp_id]]@centroids

tmpc <- lapply(seq_along(tmpc), # make one DF out of list of matrices
               function(i) {
                 data.frame(cluster=i,
                            contribution=tmpc[[i]][,"contribution"],
                            experience=tmpc[[i]][,"experience"],
                            period=2:(dim(tmpc[[i]])[1]+1))
               })


tmpc<-do.call(rbind, tmpc)
tmpc$maxPeriod <- 20


# add cluster information
clust_clust<-as.data.frame(comparison[["objects.partitional"]][[exp_id]]@cluster)
clust_clust$uid<-1:dim(clust_clust)[1]
tmp<-merge(tmp, clust_clust, by = "uid")  # add cluster information
colnames(tmp)[5]<-"cluster"

tmp$maxPeriod = 20
clust_data <- rbind(clust_data, tmp)
centr <- rbind(centr, tmpc)

clust_data = clust_data%>% group_by(maxPeriod, uid) %>%
  mutate(contrib_others_lagged_1 = experience,
         contrib_others_lagged_2 = c(NA, experience[1:(length(experience)-1)]),
         contrib_own_lagged_1 = c(NA, contribution[1:(length(contribution)-1)]),
         period_frac = round(period/max(period),2),
         contrib_frac = ifelse(is.na(contrib_own_lagged_1/contrib_others_lagged_1) | is.infinite(contrib_own_lagged_1/contrib_others_lagged_1), 
                               ifelse(contrib_others_lagged_1 == 0 & contrib_own_lagged_1 != 0, 1,0),
                               contrib_own_lagged_1/contrib_others_lagged_1),
         contrib_frac_lag1 = ifelse(is.na(contrib_own_lagged_1/contrib_others_lagged_2)| is.infinite(contrib_own_lagged_1/contrib_others_lagged_2), 
                                    ifelse(contrib_others_lagged_2 == 0 & contrib_own_lagged_1 != 0, 1,0),
                                    contrib_own_lagged_1/contrib_others_lagged_2)
  )

centr = centr%>% group_by(maxPeriod) %>%
  mutate(contrib_others_lagged_1 = experience,
         contrib_others_lagged_2 = c(NA, experience[1:(length(experience)-1)]),
         contrib_own_lagged_1 = c(NA, contribution[1:(length(contribution)-1)]),
         period_frac = round(period/max(period),2),
         contrib_frac = ifelse(is.na(contrib_own_lagged_1/contrib_others_lagged_1) | is.infinite(contrib_own_lagged_1/contrib_others_lagged_1), 
                               ifelse(contrib_others_lagged_1 == 0 & contrib_own_lagged_1 != 0, 1,0),
                               contrib_own_lagged_1/contrib_others_lagged_1),
         contrib_frac_lag1 = ifelse(is.na(contrib_own_lagged_1/contrib_others_lagged_2)| is.infinite(contrib_own_lagged_1/contrib_others_lagged_2), 
                                    ifelse(contrib_others_lagged_2 == 0 & contrib_own_lagged_1 != 0, 1,0),
                                    contrib_own_lagged_1/contrib_others_lagged_2)
  )


sub7_1 <- clust_data[clust_data$maxPeriod ==7 & clust_data$period>1,]
sub7_2 <- clust_data[clust_data$maxPeriod ==7 & clust_data$period>2,]
sub10_1 <- clust_data[clust_data$maxPeriod ==10 & clust_data$period>1,]
sub10_2 <- clust_data[clust_data$maxPeriod ==10 & clust_data$period>2,]
sub20_1 <- clust_data[clust_data$maxPeriod ==20 & clust_data$period>1,]
sub20_2 <- clust_data[clust_data$maxPeriod ==20& clust_data$period>2,]
#sub30_1 <- clust_data[clust_data$maxPeriod ==30 & clust_data$period>1,]
#sub30_2 <- clust_data[clust_data$maxPeriod ==30 & clust_data$period>2,]


###Train Grammatical Evolution on the different subgroups
subframes <- c("sub7_1", "sub7_2","sub10_1", "sub10_2", "sub20_1", "sub20_2")
depth_list <- c(3,4,5)
sublist <- list()


for(frame in subframes){
  tmp <- eval(parse(text=as.character(frame)))
  
  write.csv(tmp, file = paste('Data/results/clustering_real_data/', frame, '.txt', sep=''))
  
  
  
}

for(frame in subframes){
  tmp <- read.csv(file = paste('Data/results/clustering_real_data/', frame, '.txt', sep=''))
  do.call("<-",list(frame, tmp))
  
  
}

###Train Grammatical Evolution on the different subgroups
#subframes <- c("sub7_1", "sub7_2","sub10_1", "sub10_2", "sub20_1", "sub20_2")
subframes <- c("sub7_2","sub10_2", "sub20_2")
depth_list <- c(3,4,5)
sublist <- list()

#here we use the gramEvol package for our grammatical Evolution (we specify depths and which variables to use)
for(depth in c(4,5)){
  set.seed(12345)
  for(j in subframes){
    tmp <- eval(parse(text=j))
    ge_list <- list()
    for(c in sort(unique(tmp$cluster))){
      
      contrib_frac <- tmp$contrib_frac[tmp$cluster == c]
      contrib_own_lagged_1 <- tmp$contrib_own_lagged_1[tmp$cluster == c]
      contrib_others_lagged_1 <- tmp$contrib_others_lagged_1[tmp$cluster == c]
      period_frac <- tmp$period_frac[tmp$cluster == c]
      contrib_others_lagged_2 <- tmp$contrib_others_lagged_2[tmp$cluster == c]
      contribution <- tmp$contribution[tmp$cluster == c]
      
      if(grepl("_2", j, fixed=TRUE)){
        rules <- grule(contrib_frac,
                       contrib_own_lagged_1,
                       contrib_others_lagged_1,
                       contrib_others_lagged_2,
                       period_frac,
                       n)
        
      }else{
        rules <- grule(contrib_frac,
                       #contrib_own_lagged_1,
                       contrib_others_lagged_1,
                       period_frac,
                       n)
        
      }
      #here we specify the allowed grammar to use
      print(paste('Doing evolution for cluster', c, sep=' '))
      ruleDef <- list(expr = grule(op(expr, expr), func(expr), var),
                      func = grule(sin, exp, log, cos),
                      op = grule('+', '-', '*', '/', '^'),
                      var = rules,
                      n = grule(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))  
      
      grammarDef <- CreateGrammar(ruleDef)
      
      
      ##many possible options - define two here
      nrmse <- function(true, pred){mean(log(1+abs(true-pred)))}
      rmse <- function(true, prediction){sqrt(mean((true-prediction)^2))}
      
      SymRegFitFunc <- function(expr) {
        result <- eval(expr)
        if (any(is.nan(result))){
          return(Inf)
        }
        res <- rmse(contribution, result)
        #if(any(is.nan(res))| any(is.na(res))){
        #  print('Cost calculation failed...')
        #  #print(result)
        #  return(Inf)
        #}
        return(res)
      }
      #here we call the actual aglorithm
      set.seed(2)
      ge <- GrammaticalEvolution(grammarDef, SymRegFitFunc, max.depth=depth, terminationCost = 0.05, iterations=5000)
      print(ge)
      ge_list[[length(ge_list)+1]] <- ge
      names(ge_list)[length(ge_list)] <- as.character(c)
    }
    sublist[[length(sublist)+1]] <- ge_list
    names(sublist)[length(sublist)] <- j
    saveRDS(ge_list, file = paste('ge_new_',j,'depth_',depth,'.rds', sep=''))
  }
  
  saveRDS(sublist, file=paste("ge_all_new_depth_", depth,".rds", sep=''))
}

#################################################################################################
# Here we look at and work we specific results from above
#################################################################################################

#depth = 3
#sublist3 <- readRDS(file=paste("ge_all_depth_", depth,".rds", sep=''))
depth = 4
sublist4 <- readRDS(file=paste("ge_all_depth_", depth,".rds", sep=''))
#depth = 5
#sublist5 <- readRDS(file=paste("ge_all_depth_", depth,".rds", sep=''))

#subframes <- c("sub7_1", "sub7_2","sub10_1", "sub10_2", "sub20_1", "sub20_2") 
subframes <- c("sub7_2", "sub20_2") 
slist <- list( "sublist4")

df <- data.frame(dataset = numeric(),
                 lookback = numeric(),
                 depth = numeric(),
                 cluster = numeric(),
                 size = numeric(),
                 fun = character(),
                 cost = numeric(),
                 time = numeric(),
                 others1 = numeric(),
                 others2 = numeric(),
                 own=numeric(),
                 static=numeric(),
                 trig = numeric(),
                 exp = numeric(),
                 poly=numeric())


timen <- c("period_frac")
othersn2 <- c("contrib_others_lagged_2")
othersn1 <- c("contrib_others_lagged_1", "contrib_frac")
ownn1 <- c("contrib_frac", "contrib_own_lagged_1")
staticn <- "\\s+\\\\-\\s+[0-9]"
trign <- c("sin")
expn <- c("exp")
polyn <- c("^")

for(nm in slist){
  for(frame in subframes){
    res <- eval(parse(text=nm))
    

    for(i in names(res[[frame]])){
      
      depth = str_extract(nm, '[0-9]+')[[1]][1]
      extr <- str_extract_all(frame, "[0-9]+")[[1]]
      tmp <- clust_data[clust_data$maxPeriod == extr[[1]],]
      div <- ifelse(grepl('_2', frame), unique(tmp$maxPeriod)-2, unique(tmp$maxPeriod)-1)
      tmp <- tmp %>% group_by(maxPeriod, cluster) %>% summarise(n = (n()/div))
      
      cost <- res[[frame]][[as.character(i)]][["best"]][["cost"]]
      fun <- toString(res[[frame]][[as.character(i)]][["best"]][["expressions"]])
      nms <- all.names(res[[frame]][[as.character(i)]][["best"]][["expressions"]])
      tree <- capture.output(call_tree(res[[frame]][[as.character(i)]][["best"]][["expressions"]]))
      
      time <- ifelse(TRUE %in% (nms %in% timen), 1,0)
      others1 <- ifelse(TRUE %in% (nms %in% othersn1), 1,0)
      others2 <- ifelse(TRUE %in% (nms %in% othersn2), 1,0)
      own <- ifelse(TRUE %in% (nms %in% ownn1), 1,0)
      static <- ifelse(TRUE %in% (grepl(staticn, tree)), 1,0)
      trig <- ifelse(TRUE %in% (nms %in% trign), 1,0)
      exp <- ifelse(TRUE %in% (nms %in% expn), 1,0)
      poly <- ifelse(TRUE %in% (nms %in% polyn), 1,0)
      
      
      
      
      
      df[nrow(df)+1,] <- c(extr[1], extr[2], depth, as.numeric(i), tmp[tmp$cluster == as.numeric(i), 'n'], fun, cost,
                           time, others1, others2, own, static, trig, exp, poly)
      
      
    }
  }
  
}
write.csv(df, file="Data/results/grammatical_evolution.csv")


contrib_others_lagged_1 <- seq(0,1,0.01)
contrib_others_lagged_2 <- seq(0,1,0.01)
contrib_own_lagged_1 <- seq(0,1,0.01)
contrib_frac <- seq(0,1,0.01)
period_frac <- seq(0,1,0.01)

df$x <- "seq(0,1,0.01)"
##set parameters for subgroup
data = 7
lookback = 2
depth = 4
time = c(1)
own = c(0)
others1 = c(0)
others2 = c(0)
static = c(0,1)





##time dependent only
bool <- df$dataset == data & df$lookback == lookback & df$depth == depth & df$time %in% time &df$own %in% own & df$others1 %in% others1 & df$others2 %in% others2 & df$static %in% static


ls <- eval(parse(text=paste('sublist',depth,sep='')))


  n <- paste('sub', data,'_', lookback, sep='')
  fr <- eval(parse(text = n))
  cent <- centr[centr$maxPeriod == data,]
  ##calc results
  for(clus in unique(fr$cluster)){
    clustermask <- fr$cluster %in% c(clus)
    mask2 <- cent$cluster %in% c(clus)
    
    fr[clustermask, 'estimate'] <- EvalExpressions(ls[[n]][[as.character(clus)]]$best$expressions,
                                                   envir=fr[clustermask,])
    
    fr[clustermask, 'expr'] <-  toString(ls[[n]][[as.character(clus)]]$best$expressions)
    cent[mask2, 'expr'] <-  toString(ls[[n]][[as.character(clus)]]$best$expressions)
    cent[mask2,'estimate'] <- EvalExpressions(ls[[n]][[as.character(clus)]]$best$expressions,
                                              envir=cent[mask2,])
    
    
  }
  cent$uid <- paste('centr_', cent$cluster, sep='')
  cent$type <- 'centroid'
  fr$type <- 'observed data'
  fr$uid <- as.character(fr$uid)
  pf <- rbind(fr, cent)
  rel_col <-  c('uid','cluster','type','period','contribution','experience','estimate', 'expr')
  pf <- tidyr::gather(pf[,rel_col], "measure", "value",-period,-uid,-cluster,-type, -expr,factor_key = FALSE)
  pf$facet <- ifelse(pf$measure %in% c('estimate', 'contribution'), 'contribution', 'experience')
  pf$data_type <- ifelse(pf$measure == 'estimate', 'Estimtate',
                         ifelse(pf$type == 'centroid', 'Centroid', 'Real data Point'))
  
  k <- length(unique(pf$cluster))
  pf_1<-pf[pf$cluster %in% 1:floor(k/2) & pf$period>2,]
  pf_2<-pf[pf$cluster %in% (floor(k/2)+1):k& pf$period>2,]
  pf_1$cluster <-factor(pf_1$cluster, levels = sort(unique(pf_1$cluster)))
  pf_2$cluster <-factor(pf_2$cluster, levels = sort(unique(pf_2$cluster)))
  
  p1<- ggplot()+
    geom_line(data=pf_1[pf_1$measure != 'estimate' & pf_1$type != 'centroid',],
              mapping=aes(x=period, y=value, group=uid),col="grey")+
    geom_point(data=pf_1[(pf_1$type == 'centroid'),],
               aes(x=period, y=value, color=data_type))+
    geom_line(data=pf_1[(pf_1$type == 'centroid'),],
              aes(x=period, y=value, color=data_type))+
    scale_x_continuous(breaks = seq(3, max(pf$period), by = 1))+
    scale_y_continuous(breaks = seq(0, 1, by = 0.5))+
    scale_color_manual(name = "Type", labels = c('Centroid', 'Estimate'), values= c('#009900', '#000099'))+
    facet_grid(rows = vars(cluster), cols = vars(facet))+
    theme(#legend.position=none, 
      panel.border = element_rect(colour = "black", fill = NA),
      panel.background=element_blank(), 
      panel.grid.minor=element_blank(), 
      panel.grid.major=element_blank())
  p2<- ggplot()+
    geom_line(data=pf_2[pf_2$measure != 'estimate' & pf_2$type != 'centroid',],
              mapping=aes(x=period, y=value, group=uid),col="grey")+
    geom_point(data=pf_2[(pf_2$type == 'centroid'),],
               aes(x=period, y=value, color=data_type))+
  geom_line(data=pf_2[(pf_2$type == 'centroid'),],
            aes(x=period, y=value, color=data_type))+
    scale_x_continuous(breaks = seq(3, max(pf$period), by = 1))+
    scale_y_continuous(breaks = seq(0, 1, by = 0.5))+
    scale_color_manual(name = "Type", labels = c('Centroid', 'Estimate'), values= c('#009900', '#000099'))+
    facet_grid(rows = vars(cluster), cols = vars(facet))+
    theme(#legend.position=none, 
      panel.border = element_rect(colour = "black", fill = NA),
      panel.background=element_blank(), 
      panel.grid.minor=element_blank(), 
      panel.grid.major=element_blank())
  
  
  plot_list <- vector('list', 2)
  plot_list[[1]]<-p1
  plot_list[[2]]<-p2
  p_all<-ggarrange(plot_list[[1]], 
                   plot_list[[2]] + 
                     theme(axis.text.y = element_blank(),
                           axis.ticks.y = element_blank(),
                           axis.title.y = element_blank() ), 
                   ncol=2, nrow=1, common.legend = TRUE, legend="bottom", widths = c(1,1,1))
  p_all
  ggsave(plot=p_all, filename = paste("../figures/estimated_by_time_",7,".pdf", sep=""), height=10)

  
  ##functional plots from here
  
  
contrib_others_lagged_1 <- seq(0,1,0.01)
contrib_others_lagged_2 <- seq(0,1,0.01)
contrib_own_lagged_1 <- seq(0,1,0.01)
contrib_frac <- seq(0,1,0.01)
period_frac <- seq(0,1,0.01)
  
df$x <- "seq(0,1,0.01)"
##set parameters for subgroup
data = 7
lookback = 2
depth = 4
time = c(1)
own = c(0)
others1 = c(0)
others2 = c(0)
static = c(0,1) 

#time dependent only
bool <- df$dataset == data & df$lookback == lookback & df$depth == depth & df$time %in% time &df$own %in% own & df$others1 %in% others1 & df$others2 %in% others2 & df$static %in% static

sub <- df[bool,]
vars <- c('period_frac')
x<-c()
y<-c()
cluster <- c()
range <- c()
for(i in sub$cluster){
  
  tmp <- data.frame(x=eval(parse(text=df[bool &df$cluster == i,"x"])),
                    y=eval(parse(text=df[bool &df$cluster == i,"fun"])),
                    cluster=i, fun = df[bool &df$cluster == i,"fun"],
                    typ = 'estimated function')
  tmp[tmp$y >1,'y'] <-1
  if(length(range)==0){
    range <- tmp
  }else{
    range <- rbind(range, tmp)
  }
  
  
  
}

  
ggplot()+
  geom_point(data=clust_data[clust_data$cluster %in% unique(range$cluster) & clust_data$maxPeriod == data,],
             aes(x=eval(parse(text = vars[1])), y=contribution, shape=as.factor(cluster)))+
  geom_line(data=range, aes(x=x, y=y, color = fun),size=1.5)+
  geom_point(data=range[range$x %in% seq(0,1,0.1),], aes(x=x, y=y, shape=as.factor(cluster)),size=3)+
  scale_color_brewer(name='Estimated Function', palette="Set1")+
  scale_shape(name='Cluster')+
  scale_y_continuous(name='contribution')+
  scale_y_continuous(name='current period/# periods')+
  theme(legend.position="bottom",
    panel.border = element_rect(colour = "black", fill = NA),
    panel.background=element_blank(), 
    panel.grid.minor=element_blank(), 
    panel.grid.major=element_blank())
  
  



#own dependent only
time = c(0)
own = c(1)
bool <- df$dataset == data & df$lookback == lookback & df$depth == depth & df$time %in% time &df$own %in% own & df$others1 %in% others1 & df$others2 %in% others2 & df$static %in% static

sub <- df[bool,]
vars <- c('contrib_own_lagged_1')

range <- c()
for(i in sub$cluster){
  
  tmp <- data.frame(x=eval(parse(text=df[bool &df$cluster == i,"x"])),
                    y=eval(parse(text=df[bool &df$cluster == i,"fun"])),
                    cluster=i, fun = df[bool &df$cluster == i,"fun"],
                    typ = 'estimated function')
  tmp[tmp$y >1,'y'] <-1
  if(length(range)==0){
    range <- tmp
  }else{
    range <- rbind(range, tmp)
  }
  
  
  
}


ggplot()+
  geom_point(data=clust_data[clust_data$cluster %in% unique(range$cluster) & clust_data$maxPeriod == data,],
             aes(x=eval(parse(text = vars[1])), y=contribution, shape=as.factor(cluster)))+
  geom_line(data=range, aes(x=x, y=y, color = fun),size=1.)+
  geom_point(data=range[range$x %in% seq(0,1,0.1),], aes(x=x, y=y, shape=as.factor(cluster)),size=3)+
  scale_color_brewer(name='Estimated Function', palette="Set1")+
  scale_shape(name='Cluster')+
  scale_y_continuous(name='contribution')+
  scale_x_continuous(name='own contribution last round')+
  theme(legend.position="bottom",
        legend.box="vertical", legend.margin=margin(),
        panel.border = element_rect(colour = "black", fill = NA),
        panel.background=element_blank(), 
        panel.grid.minor=element_blank(), 
        panel.grid.major=element_blank())



#own dependent only
others1 = c(1)
own = c(0)
bool <- df$dataset == data & df$lookback == lookback & df$depth == depth & df$time %in% time &df$own %in% own & df$others1 %in% others1 & df$others2 %in% others2 & df$static %in% static

sub <- df[bool,]
vars <- c('contrib_others_lagged_1')

range <- c()
for(i in sub$cluster){
  
  tmp <- data.frame(x=eval(parse(text=df[bool &df$cluster == i,"x"])),
                    y=eval(parse(text=df[bool &df$cluster == i,"fun"])),
                    cluster=i, fun = df[bool &df$cluster == i,"fun"],
                    typ = 'estimated function')
  tmp[tmp$y >1,'y'] <- 0
  if(length(range)==0){
    range <- tmp
  }else{
    range <- rbind(range, tmp)
  }
  
  
  
}


ggplot()+
  geom_point(data=clust_data[clust_data$cluster %in% unique(range$cluster) & clust_data$maxPeriod == data,],
             aes(x=eval(parse(text = vars[1])), y=contribution, shape=as.factor(cluster)))+
  geom_line(data=range, aes(x=x, y=y, color = fun),size=1.5)+
  geom_point(data=range[range$x %in% seq(0,1,0.1),], aes(x=x, y=y, shape=as.factor(cluster)),size=3)+
  scale_color_brewer(name='Estimated Function', palette="Set1")+
  scale_shape(name='Cluster')+
  scale_y_continuous(name='contribution')+
  scale_x_continuous(name='avg contribution of others last round')+
  theme(legend.position="bottom",
        legend.box="vertical", legend.margin=margin(),
        panel.border = element_rect(colour = "black", fill = NA),
        panel.background=element_blank(), 
        panel.grid.minor=element_blank(), 
        panel.grid.major=element_blank())



#own dependent only
others1 = c(0)
others2 = c(1)
bool <- df$dataset == data & df$lookback == lookback & df$depth == depth & df$time %in% time &df$own %in% own & df$others1 %in% others1 & df$others2 %in% others2 & df$static %in% static

sub <- df[bool,]
vars <- c('contrib_others_lagged_2')

range <- c()
for(i in sub$cluster){
  
  tmp <- data.frame(x=eval(parse(text=df[bool &df$cluster == i,"x"])),
                    y=eval(parse(text=df[bool &df$cluster == i,"fun"])),
                    cluster=i, fun = df[bool &df$cluster == i,"fun"],
                    typ = 'estimated function')
  tmp[tmp$y >1,'y'] <-1
  if(length(range)==0){
    range <- tmp
  }else{
    range <- rbind(range, tmp)
  }
  
  
  
}


ggplot()+
  geom_point(data=clust_data[clust_data$cluster %in% unique(range$cluster) & clust_data$maxPeriod == data,],
             aes(x=eval(parse(text = vars[1])), y=contribution, shape=as.factor(cluster)))+
  geom_line(data=range, aes(x=x, y=y, color = fun),size=1.5)+
  geom_point(data=range[range$x %in% seq(0,1,0.1),], aes(x=x, y=y, shape=as.factor(cluster)),size=3)+
  scale_color_brewer(name='Estimated Function', palette="Set1")+
  scale_shape(name='Cluster')+
  scale_y_continuous(name='contribution')+
  scale_x_continuous(name='avg contribution of others two rounds before')+
  theme(legend.position="bottom",
        legend.box="vertical", legend.margin=margin(),
        panel.border = element_rect(colour = "black", fill = NA),
        panel.background=element_blank(), 
        panel.grid.minor=element_blank(), 
        panel.grid.major=element_blank())

library("plot3D")
library(lattice)
#own dependent and other dependent
others1 = c(1)
others2 = c(0)
own=c(1)
x <- tidyr::expand_grid(x=seq(0,1,0.01), y=seq(0,1,0.01))
contrib_others_lagged_1 <- x$x
contrib_others_lagged_2 <- seq(0,1,0.01)
contrib_own_lagged_1 <- x$y
contrib_frac <- x$x/x$y
contrib_frac <- ifelse(contrib_frac >1, 1, contrib_frac)
period_frac <- seq(0,1,0.01)
bool <- df$dataset == data & df$lookback == lookback & df$depth == depth & df$time %in% time &df$own %in% own & df$others1 %in% others1 & df$others2 %in% others2 & df$static %in% static

sub <- df[bool,]
vars <- c('contrib_others_lagged_2')


rr<-c(0,1)
for(i in sub$cluster){
  
  tmp <- data.frame(x=eval(parse(text=df[bool &df$cluster == i,"x"])),
                    y=eval(parse(text=df[bool &df$cluster == i,"x"])),
                    z=eval(parse(text=df[bool &df$cluster == i,"fun"])),
                    cluster=i, fun = df[bool &df$cluster == i,"fun"],
                    typ = 'estimated function')
  tmp[tmp$y >1,'y'] <-1
  
  c <- cloud(z~x+y, tmp, xlim=rr, ylim=rr, zlim=rr, alpha.regions = 1);
  print(c)

}




######################## Repeat for 20 Periods ##################################  
###20 periods
contrib_others_lagged_1 <- seq(0,1,0.01)
contrib_others_lagged_2 <- seq(0,1,0.01)
contrib_own_lagged_1 <- seq(0,1,0.01)
contrib_frac <- seq(0,1,0.01)
period_frac <- seq(0,1,0.01)  

data <- 20
  
  n <- paste('sub', data,'_', lookback, sep='')
  fr <- eval(parse(text = n))
  cent <- centr[centr$maxPeriod == data,]
  ##calc results
  for(clus in unique(fr$cluster)){
    clustermask <- fr$cluster %in% c(clus)
    mask2 <- cent$cluster %in% c(clus)
    
    fr[clustermask, 'estimate'] <- EvalExpressions(ls[[n]][[as.character(clus)]]$best$expressions,
                                                   envir=fr[clustermask,])
    
    fr[clustermask, 'expr'] <-  toString(ls[[n]][[as.character(clus)]]$best$expressions)
    cent[mask2, 'expr'] <-  toString(ls[[n]][[as.character(clus)]]$best$expressions)
    cent[mask2,'estimate'] <- EvalExpressions(ls[[n]][[as.character(clus)]]$best$expressions,
                                              envir=cent[mask2,])
    
    
  }
  cent$uid <- paste('centr_', cent$cluster, sep='')
  cent$type <- 'centroid'
  fr$type <- 'observed data'
  fr$uid <- as.character(fr$uid)
  pf <- rbind(fr, cent)
  rel_col <-  c('uid','cluster','type','period','contribution','experience','estimate', 'expr')
  pf <- tidyr::gather(pf[,rel_col], "measure", "value",-period,-uid,-cluster,-type, -expr,factor_key = FALSE)
  pf$facet <- ifelse(pf$measure %in% c('estimate', 'contribution'), 'contribution', 'experience')
  pf$data_type <- ifelse(pf$measure == 'estimate', 'Estimtate',
                         ifelse(pf$type == 'centroid', 'Centroid', 'Real data Point'))
  
  k <- length(unique(pf$cluster))
  pf_1<-pf[pf$cluster %in% 1:floor(k/2) & pf$period>2,]
  pf_2<-pf[pf$cluster %in% (floor(k/2)+1):k& pf$period>2,]
  pf_1$cluster <-factor(pf_1$cluster, levels = sort(unique(pf_1$cluster)))
  pf_2$cluster <-factor(pf_2$cluster, levels = sort(unique(pf_2$cluster)))
  
  p1<- ggplot()+
    geom_line(data=pf_1[pf_1$measure != 'estimate' & pf_1$type != 'centroid',],
              mapping=aes(x=period, y=value, group=uid),col="grey")+
    geom_point(data=pf_1[(pf_1$type == 'centroid'),],
               aes(x=period, y=value, color=data_type))+
    geom_line(data=pf_1[(pf_1$type == 'centroid'),],
              aes(x=period, y=value, color=data_type))+
    scale_x_continuous(breaks = seq(3, max(pf$period), by = 1))+
    scale_y_continuous(breaks = seq(0, 1, by = 0.5))+
    scale_color_manual(name = "Type", labels = c('Centroid', 'Estimate'), values= c('#009900', '#000099'))+
    facet_grid(rows = vars(cluster), cols = vars(facet))+
    theme(#legend.position=none, 
      panel.border = element_rect(colour = "black", fill = NA),
      panel.background=element_blank(), 
      panel.grid.minor=element_blank(), 
      panel.grid.major=element_blank())
  p2<- ggplot()+
    geom_line(data=pf_2[pf_2$measure != 'estimate' & pf_2$type != 'centroid',],
              mapping=aes(x=period, y=value, group=uid),col="grey")+
    geom_point(data=pf_2[(pf_2$type == 'centroid'),],
               aes(x=period, y=value, color=data_type))+
    geom_line(data=pf_2[(pf_2$type == 'centroid'),],
              aes(x=period, y=value, color=data_type))+
    scale_x_continuous(breaks = seq(3, max(pf$period), by = 1))+
    scale_y_continuous(breaks = seq(0, 1, by = 0.5))+
    scale_color_manual(name = "Type", labels = c('Centroid', 'Estimate'), values= c('#009900', '#000099'))+
    facet_grid(rows = vars(cluster), cols = vars(facet))+
    theme(#legend.position=none, 
      panel.border = element_rect(colour = "black", fill = NA),
      panel.background=element_blank(), 
      panel.grid.minor=element_blank(), 
      panel.grid.major=element_blank())
  
  
  plot_list <- vector('list', 2)
  plot_list[[1]]<-p1
  plot_list[[2]]<-p2
  p_all<-ggarrange(plot_list[[1]], 
                   plot_list[[2]] + 
                     theme(axis.text.y = element_blank(),
                           axis.ticks.y = element_blank(),
                           axis.title.y = element_blank() ), 
                   ncol=2, nrow=1, common.legend = TRUE, legend="bottom", widths = c(1,1,1))
  p_all
  ggsave(plot=p_all, filename = paste("../figures/estimated_by_time_",data,".pdf", sep=""), height=10)
  
  ####plot selected clusters 17, 11, 42, 4
  
  for(i in c(4,11,17,42)){
    p <- ggplot()+
    geom_line(data=pf[pf$measure != 'estimate' & pf$type != 'centroid' & pf$cluster == i,],
              mapping=aes(x=period, y=value, group=uid),col="grey")+
      geom_point(data=pf[(pf$type == 'centroid' & pf$cluster == i),],
                 aes(x=period, y=value, color=data_type))+
      geom_line(data=pf[(pf$type == 'centroid' & pf$cluster == i),],
                aes(x=period, y=value, color=data_type))+
      scale_x_continuous(name = 'period', breaks = seq(3, max(pf$period), by = 1))+
      scale_y_continuous(name='normed value', breaks = seq(0, 1, by = 0.5),limits=c(0,1), expand = c(0,0))+
      scale_color_manual(name = "Type", labels = c('Centroid',
                                                   paste('Fun=',df[df$dataset == data &df$cluster ==i, 'fun'], sep=' ')),
                         values= c('#009900', '#000099'))+
      facet_grid(cols = vars(facet))+
      theme(legend.position='bottom',
        panel.border = element_rect(colour = "black", fill = NA),
        panel.background=element_blank(), 
        panel.grid.minor=element_blank(), 
        panel.grid.major=element_blank())
    
    ggsave(plot=p, filename = paste("../figures/estimated_by_time_",data,'_cluster_',i,".pdf", sep=""), height=4, width=10)
    
  }
  
  
  
  ##functional plots from here
  
  
  contrib_others_lagged_1 <- seq(0,1,0.01)
  contrib_others_lagged_2 <- seq(0,1,0.01)
  contrib_own_lagged_1 <- seq(0,1,0.01)
  contrib_frac <- seq(0,1,0.01)
  period_frac <- seq(0,1,0.01)
  
  df$x <- "seq(0,1,0.01)"
  ##set parameters for subgroup
  lookback = 2
  depth = 4
  time = c(1)
  own = c(0)
  others1 = c(0)
  others2 = c(0)
  static = c(0,1) 
  
  #time dependent only
  bool <- df$dataset == data & df$lookback == lookback & df$depth == depth & df$time %in% time &df$own %in% own & df$others1 %in% others1 & df$others2 %in% others2 & df$static %in% static
  
  sub <- df[bool,]
  vars <- c('period_frac')
  x<-c()
  y<-c()
  cluster <- c()
  range <- c()
  for(i in sub$cluster){
    
    tmp <- data.frame(x=eval(parse(text=df[bool &df$cluster == i,"x"])),
                      y=eval(parse(text=df[bool &df$cluster == i,"fun"])),
                      cluster=i,
                      fun = df[bool &df$cluster == i,"fun"],
                      fun_error = paste(df[bool &df$cluster == i,"fun"],' ',
                                        ' (err: ', round(df[bool&df$cluster %in% i, 'cost'],2),')',sep = ''),
                      typ = 'estimated function')
    tmp[tmp$y >1,'y'] <-1
    if(length(range)==0){
      range <- tmp
    }else{
      range <- rbind(range, tmp)
    }
    
    
    
  }
  
  
  p <- ggplot()+
    geom_point(data=clust_data[clust_data$cluster %in% unique(range$cluster),],
               aes(x=eval(parse(text = vars[1])), y=contribution, shape=as.factor(cluster)))+
    geom_line(data=range, aes(x=x, y=y, color = fun_error),size=1.5)+
    geom_point(data=range[range$x %in% seq(0,1,0.1),], aes(x=x, y=y, shape=as.factor(cluster)),size=3)+
    scale_color_brewer(name='Estimated Function', palette="Set1",
                       guide=guide_legend(ncol = 3,
                                          nrow = 2))+
    scale_shape(name='Cluster')+
    scale_y_continuous(name='contribution')+
    scale_x_continuous(name='current period/# periods')+
    theme(legend.position="bottom",
          legend.box="vertical", 
          panel.border = element_rect(colour = "black", fill = NA),
          panel.background=element_blank(), 
          panel.grid.minor=element_blank(), 
          panel.grid.major=element_blank())
  
  p
  ggsave(plot=p, filename = paste("../figures/gram_evol_",data,"_time.pdf", sep=""), height=10, width=20)
  
  
  
  #own dependent only
  time = c(0)
  own = c(1)
  bool <- df$dataset == data & df$lookback == lookback & df$depth == depth & df$time %in% time &df$own %in% own & df$others1 %in% others1 & df$others2 %in% others2 & df$static %in% static
  
  sub <- df[bool,]
  vars <- c('contrib_own_lagged_1')
  
  range <- c()
  for(i in sub$cluster){
    
    tmp <- data.frame(x=eval(parse(text=df[bool &df$cluster == i,"x"])),
                      y=eval(parse(text=df[bool &df$cluster == i,"fun"])),
                      cluster=i, fun = df[bool &df$cluster == i,"fun"],
                      fun_error = paste(df[bool &df$cluster == i,"fun"],' ',
                                        ' (err: ', round(df[bool&df$cluster %in% i, 'cost'],2),')',sep = ''),
                      typ = 'estimated function')
    tmp[tmp$y >1,'y'] <-1
    if(length(range)==0){
      range <- tmp
    }else{
      range <- rbind(range, tmp)
    }
    
    
    
  }
  
  
  p<- ggplot()+
    geom_point(data=clust_data[clust_data$cluster %in% unique(range$cluster),],
               aes(x=eval(parse(text = vars[1])), y=contribution, shape=as.factor(cluster)))+
    geom_line(data=range, aes(x=x, y=y, color = fun_error),size=1.)+
    geom_point(data=range[range$x %in% seq(0,1,0.1),], aes(x=x, y=y, shape=as.factor(cluster)),size=3)+
    scale_color_brewer(name='Estimated Function', palette="Set1",
                       guide=guide_legend(ncol = 3,
                                          nrow = 2))+
    scale_shape(name='Cluster')+
    scale_y_continuous(name='contribution')+
    scale_x_continuous(name='own contribution last round')+
    theme(legend.position="bottom",
          legend.box="vertical", legend.margin=margin(),
          panel.border = element_rect(colour = "black", fill = NA),
          panel.background=element_blank(), 
          panel.grid.minor=element_blank(), 
          panel.grid.major=element_blank())
  
  p
  ggsave(plot=p, filename = paste("../figures/gram_evol_",data,"_contrib_own.pdf", sep=""), height=10, width=20)
  
  #others1 dependent only
  others1 = c(1)
  own = c(0)
  bool <- df$dataset == data & df$lookback == lookback & df$depth == depth & df$time %in% time &df$own %in% own & df$others1 %in% others1 & df$others2 %in% others2 & df$static %in% static
  
  sub <- df[bool,]
  vars <- c('contrib_others_lagged_1')
  
  range <- c()
  for(i in sub$cluster){
    if(round(df[bool&df$cluster %in% i, 'cost'],2) <=0.15){
      tmp <- data.frame(x=eval(parse(text=df[bool &df$cluster == i,"x"])),
                        y=eval(parse(text=df[bool &df$cluster == i,"fun"])),
                        cluster=i, fun = df[bool &df$cluster == i,"fun"],
                        fun_error = paste(df[bool &df$cluster == i,"fun"],' ',
                                          ' (err: ', round(df[bool&df$cluster %in% i, 'cost'],2),')',sep = ''),
                        typ = 'estimated function')
      tmp[tmp$y >1,'y'] <-1
      if(length(range)==0){
        range <- tmp
      }else{
        range <- rbind(range, tmp)
      }
    }
    
    
  }
  
  myPalette <- colorRampPalette(brewer.pal(8, "Set1"))(length(unique(range$fun_error)))
  p <- ggplot()+
    geom_point(data=clust_data[clust_data$cluster %in% unique(range$cluster),],
               aes(x=eval(parse(text = vars[1])), y=contribution, color=as.factor(cluster)))+
    geom_line(data=range, aes(x=x, y=y, linetype= fun_error),size=1)+
    geom_point(data=range[range$x %in% seq(0,1,0.1),], aes(x=x, y=y, color=as.factor(cluster)),size=3)+
    scale_color_manual(name='Cluster', values = myPalette,
                       guide=guide_legend(ncol = 5,
                                          nrow = 3))+
    scale_linetype(name='Estimated Funtion',
                   guide=guide_legend(ncol = 3,
                                      nrow = 3))+
    scale_y_continuous(name='contribution')+
    scale_x_continuous(name='avg contribution of others last round')+
    theme(legend.position="bottom",
          legend.box="vertical", legend.margin=margin(),
          panel.border = element_rect(colour = "black", fill = NA),
          panel.background=element_blank(), 
          panel.grid.minor=element_blank(), 
          panel.grid.major=element_blank())
  p
  ggsave(plot=p, filename = paste("../figures/gram_evol_",data,"_others1.pdf", sep=""), height=10, width=20)
  
  #others2 dependent only
  others1 = c(0)
  others2 = c(1)
  bool <- df$dataset == data & df$lookback == lookback & df$depth == depth & df$time %in% time &df$own %in% own & df$others1 %in% others1 & df$others2 %in% others2 & df$static %in% static
  
  sub <- df[bool,]
  vars <- c('contrib_others_lagged_2')
  
  range <- c()
  for(i in sub$cluster){
    
    tmp <- data.frame(x=eval(parse(text=df[bool &df$cluster == i,"x"])),
                      y=eval(parse(text=df[bool &df$cluster == i,"fun"])),
                      cluster=i, fun = df[bool &df$cluster == i,"fun"],
                      fun_error = paste(df[bool &df$cluster == i,"fun"],' ',
                                        ' (err: ', round(df[bool&df$cluster %in% i, 'cost'],2),')',sep = ''),
                      typ = 'estimated function')
    tmp[tmp$y >1,'y'] <-1
    if(length(range)==0){
      range <- tmp
    }else{
      range <- rbind(range, tmp)
    }
    
    
    
  }
  
  
  p<-ggplot()+
    geom_point(data=clust_data[clust_data$cluster %in% unique(range$cluster),],
               aes(x=eval(parse(text = vars[1])), y=contribution, shape=as.factor(cluster)))+
    geom_line(data=range, aes(x=x, y=y, color = fun_error),size=1.5)+
    geom_point(data=range[range$x %in% seq(0,1,0.1),], aes(x=x, y=y, shape=as.factor(cluster)),size=3)+
    scale_color_brewer(name='Estimated Function', palette="Set1",
                       guide=guide_legend(ncol = 3,
                                          nrow = 2))+
    scale_shape(name='Cluster')+
    scale_y_continuous(name='contribution')+
    scale_x_continuous(name='avg contribution of others two rounds before')+
    theme(legend.position="bottom",
          legend.box="vertical", legend.margin=margin(),
          panel.border = element_rect(colour = "black", fill = NA),
          panel.background=element_blank(), 
          panel.grid.minor=element_blank(), 
          panel.grid.major=element_blank()) 
  
  p
  ggsave(plot=p, filename = paste("../figures/gram_evol_",data,"_others2.pdf", sep=""), height=10, width=20)

  library("plot3D")
  library(lattice)
  library(latticeExtra)
  #own dependent and other dependent
  others1 = c(1)
  others2 = c(0)
  own=c(1)
  x <- tidyr::expand_grid(x=seq(0,1,0.01), y=seq(0,1,0.01))
  contrib_others_lagged_1 <- x$x
  contrib_others_lagged_2 <- seq(0,1,0.01)
  contrib_own_lagged_1 <- x$y
  contrib_frac <- x$x/x$y
  contrib_frac <- ifelse(is.infinite(contrib_frac), 0, contrib_frac)
  contrib_frac <- ifelse(contrib_frac >1, 1, contrib_frac)
  period_frac <- seq(0,1,0.01)
  bool <- df$dataset == data & df$lookback == lookback & df$depth == depth & df$time %in% time &df$own %in% own & df$others1 %in% others1 & df$others2 %in% others2 & df$static %in% static
  
  sub <- df[bool,]
  vars <- c('contrib_others_lagged_1', 'contrib_own_lagged_1')
  
  rp <- c()
  rr<-c(0,1)
  for(i in sub$cluster){
    
    tmp <- data.frame(x=eval(parse(text=df[bool &df$cluster %in% i,"x"])),
                      y=eval(parse(text=df[bool &df$cluster %in% i,"x"])),
                      z=eval(parse(text=df[bool &df$cluster %in% i,"fun"])),
                      cluster=i,
                      clus_error = paste('Cluster ', i, ' (', round(df[bool&df$cluster %in% i, 'cost'],2),')',sep = ''),
                      fun = df[bool &df$cluster == i,"fun"],
                      typ = 'estimated function')
    tmp[is.infinite(tmp$y), 'y'] <-0
    tmp[tmp$y >1,'y'] <-0
    if(length(rp)>0){
      rp <- rbind(rp, tmp)
    }else{
      rp <- tmp
    }

    
  }
  c <- wireframe(z~x+y|clus_error, rp, xlim=rr, ylim=rr, zlim=rr, alpha.regions = 1,
                 shade=TRUE,
                 par.strip.text=list(cex=0.8),
                 zlab = list("contribution", rot = 90, fontsize=6),
                 xlab = list("contribution\n others\n(lag 1)", rot = 30, fontsize=6),
                 ylab = list("own\ncontribution\n(lag 1)", rot = -45, fontsize=6))
  print(c)
  trellis.device(device="pdf", file="../figures/20_periods_others_own_1.pdf", height=10, width=10)
  print(c)
  dev.off()
  
  
  
  
  