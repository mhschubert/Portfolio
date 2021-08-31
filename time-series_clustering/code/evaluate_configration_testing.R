#@Author: Marcel H. Schubert
#script evaluates the simulation result and generates plots

#!/usr/bin/env Rscript
setwd('./code')
require(dplyr)
require(ggplot2)
require(ggthemes)
require(scales)
require(wesanderson)
source("utils/helpers.R")


##load data tables
df_comp <- readRDS('../Data/simulation/gridsearch_results.rds')
df_comp[is.na(df_comp$gamma_distance), 'gamma_distance'] <- 0
df_comp[is.na(df_comp$method), 'method'] <- 'partitional'

df_eval <- readRDS('../Data/simulation/df_eval_fnd_confg_variable_grid_multivar_TRUE_internal_TRUE_k_5-54_grp_types_12345_grpsize_4-4_numrounds_10-10_endowment_20-20_n_1-6_sigma_0.6-0.9_eta_0.6-0.9_window_1-3_ind_groups_FALSE.rds')
df_reeval <- readRDS(file = '../Data/simulation/reevaluation_gridsearch.rds')
df_reeval <- df_reeval[, 1:60]
df_reeval[is.na(df_reeval$gamma_distance), 'gamma_distance'] <- 0

#in case there are factors in data
df_reeval$sigma <- as.numeric(as.character(ifelse(df_reeval$sigma == 1, 0.6, df_reeval$sigma)))
df_reeval$sigma <- as.numeric(as.character(ifelse(df_reeval$sigma == 2, 0.7, df_reeval$sigma)))
df_reeval$sigma <- as.numeric(as.character(ifelse(df_reeval$sigma == 3, 0.8, df_reeval$sigma)))
df_reeval$sigma <- as.numeric(as.character(ifelse(df_reeval$sigma == 4, 0.9, df_reeval$sigma)))

df_reeval$eta <- as.numeric(as.character(ifelse(df_reeval$eta == 1, 0.6, df_reeval$eta)))
df_reeval$eta <- as.numeric(as.character(ifelse(df_reeval$eta == 2, 0.7, df_reeval$eta)))
df_reeval$eta <- as.numeric(as.character(ifelse(df_reeval$eta == 3, 0.8, df_reeval$eta)))
df_reeval$eta <- as.numeric(as.character(ifelse(df_reeval$eta == 4, 0.9, df_reeval$eta)))

df_reeval$endowment <- ifelse(df_reeval$endowment == 1, 20, df_reeval$endowment)
df_reeval$numrounds <- ifelse(df_reeval$numrounds == 1, 10, df_reeval$endowment)
df_reeval$grpsize <- ifelse(df_reeval$grpsize == 1, 4, df_reeval$grpsize)


table(df_eval$config, dnn = c('algorithm'))
table(df_eval$config, df_eval$k, dnn = c('algorithm', '#k'))
round(prop.table(table(df_eval$config, df_eval$k, dnn = c('algorithm', '#k')), margin = 2), 2)
round(prop.table(table(df_eval$config, df_eval$distance, dnn = c('algorithm', '#distance')), margin = 1), 2)
round(prop.table(table(df_eval$config, dnn = c('algorithm'))), 2)

ss <- df_reeval[df_reeval$pick_success == "gamma_all_partitional_Sil-D-COP-DB-DBstar-CH-SF",]
round(prop.table(table(ss$centroid, ss$distance, dnn = c('centroid', '#distance')), margin = 2), 2)
round(prop.table(table(ss$centroid, ss$k, dnn = c('centroid', '#k')), margin = 2), 2)
round(prop.table(table(ss$centroid, ss$norm_centroid, dnn = c('centroid', 'norm')), margin = 1), 2)
round(prop.table(table(ss$distance, ss$norm_distance, dnn = c('centroid', 'norm')), margin = 1), 2)


round(prop.table(table(ss$gamma_distance, ss$k, dnn = c('gamma', '#k')), margin = 2), 2)

##all results - lets see what is suggested as best
cvis <- c("Sil","D", "COP","DB","DBstar","CH" , "SF")
normedn <- paste(cvis, 'normed', sep='_')
ranknames <- paste(normedn, 'rank', sep='_')


df_normed <- norm_cvis(df_comp)
df_normed <- rank_cvi(df_normed)
df_normed$rank_sum <- apply(df_normed[,ranknames], MARGIN = 1, FUN = sum)
ind <- which(df_normed$rank_sum == max(df_normed$rank_sum))
df_normed[ind, ]

#################  MEAN  and MEDIAN#################

for(i in list(mean, median)){
  ##now we do it along this grouping and then plot the groups
  along <- c('algo', 'method', 'gamma_distance', 'k')#, 'window.size_distance', 'k')
  res <- ungroup(calc_along(df_comp, along, rank=FALSE, FUN=i))
  fn <- get_fun(i)
  #make selected cols and now calculate rank along whole group over all k
  selcols <- paste(normedn, fn, sep='_')
  along <- c('algo', 'method', 'gamma_distance')#, 'window.size_distance')
  
  res <- res %>% group_by_at(along) %>% group_modify(~rank_cvi(df=.x, cvicols = selcols))
  rankcols <- paste(selcols, 'rank', sep='_')
  res$rank_sum <- apply(res[,rankcols], MARGIN = 1, FUN = sum)
  res <- res %>% group_by_at(along) %>% mutate(rank_sum = (rank_sum-min(rank_sum))/(max(rank_sum)-min(rank_sum)))
  
  #make identifier
  #res$ident <- paste(res$algo, res$method, res$gamma_distance, res$window.size_distance, sep='_')
  res$ident <- paste(res$algo, res$method, res$gamma_distance, sep='_')
  res <- tidyr::gather(res, "measure", "value",-algo, -method, -gamma_distance, -k,-ident, factor_key = TRUE)
  #norm rank_sum for scale reasons
  mask <- res$measure %in% rankcols
  res[mask, ] <- res[mask,] %>% group_by(measure) %>%
                    mutate(value = (value/min(value))/(max(value)-min(value)))
  
  res$measure <- stringr::str_remove(res$measure, '_normed')
  
  p <- ggplot(res, aes())+
    geom_line(aes(x=k, y=value, color=as.factor(gamma_distance)))+
    facet_grid(rows=vars(measure), cols=vars(method))+
    scale_color_discrete(name= 'Gamma')+
    scale_y_continuous(name='Normed Value')+
    scale_x_continuous(name='# Clusters (k)')+
    ggtitle(paste('Results over K grouped by', paste(along, collapse='/'), sep='_'))+
    theme(legend.position = 'bottom')
  
  p
  
  ggsave(p, filename = paste('../figures/grid_search/all_points_method_measure_', fn, '.pdf', sep=''),
                             height = 20, width=20)
  
  p <- ggplot(res[res$method == 'partitional',], aes())+
    geom_line(aes(x=k, y=value, color=as.factor(gamma_distance)))+
    facet_wrap(.~measure)+
    scale_color_discrete(name= 'Gamma')+
    scale_y_continuous(name='Normed Value')+
    scale_x_continuous(name='# Clusters (k)')+
    ggtitle(paste('Results over K grouped by', paste(along, collapse='/'), sep='_'))+
    theme(legend.position = 'bottom')
  
  p
  
  ggsave(p, filename = paste('../figures/grid_search/all_points_paritional_measure_', fn, '.pdf', sep=''),
         height = 10, width=15)

}

##make plot only for the best picks for each k

gammas <- sort(unique(df_reeval$gamma_distance))
gammas <- gammas[2:length(gammas)]
algos <- unique(df_reeval$config)

ident <- c()
for(a in algos){
  for(g in gammas){
  ident[length(ident)+1] <- paste("gamma_", g, "_", a, "_Sil-D-COP-DB-DBstar-CH-SF", sep='')
  }
}


sub <- df_reeval[df_reeval$pick_success %in% ident,]
sub$grps <- paste(sub$sigma, sub$eta, sub$grpsize, sub$numrounds, sub$endowment, sub$n, sub$windowsize,sep='_')
#numrounds, grpsize, endowment does not vary
sub$labs <- paste('sigma:', sub$sigma, '_eta:', sub$eta, '_window:', sub$windowsize, '_n:', sub$n, sep='')
sub$eta_sigma <- paste(sub$eta, sub$sigma, sep='_')
sub$windowsize_n <- paste(sub$windowsize, sub$n, sep='_')
##make ranking within groups across k (no k in grouping; there is only one observation for each k, i.e. the best pick)
sub <- sub %>% group_by(pick_success, grps) %>% group_modify(~rank_results(.x))

#normed rank_sum
sub$rank_sum <- apply(sub[,ranknames], MARGIN = 1, FUN = sum)
sub <- sub %>% group_by(pick_success, grps) %>% mutate(rank_sum = (rank_sum-min(rank_sum))/(max(rank_sum)-min(rank_sum)))

#order ascending in sigma and eta
sub <- sub[order(sub$grps),]

#plot for n=6
p <- ggplot(sub[grepl('_6_', sub$grps, fixed=TRUE),], aes(x=k, y=rank_sum, color=config))+
  geom_line()+
  scale_y_continuous(name='Normed Rank Sum over all CVIs')+
  scale_x_continuous(name='# Clusters (k)')+
  scale_color_discrete('Algorithm')+
  ggtitle('Rank Sum of best algorithmic configuration within dataset variations (y) and gamma (x) over k with n=6')+
  facet_grid(cols=vars(gamma_distance), rows=vars(labs))+
  theme(strip.text.y.right = element_text(angle = 0))
  
p

ggsave(filename = '../figures/grid_search/rank_sum_within_configuration_n=6.pdf', p, height = 30, width = 20)

#plot for n=3
p <- ggplot(sub[grepl('_3_', sub$grps, fixed=TRUE),], aes(x=k, y=rank_sum, color=config))+
  geom_line()+
  scale_y_continuous(name='Normed Rank Sum over all CVIs')+
  scale_x_continuous(name='# Clusters (k)')+
  scale_color_discrete('Algorithm')+
  ggtitle('Rank Sum of best algorithmic configuration within dataset variations (y) and gamma (x) over k with n=3')+
  facet_grid(cols=vars(gamma_distance), rows=vars(labs))+
  theme(strip.text.y.right = element_text(angle = 0))

p

ggsave(filename = '../figures/grid_search/rank_sum_within_configuration_n=3.pdf', p, height = 30, width = 20)


#plot for n=1
p <- ggplot(sub[grepl('_1_', sub$grps, fixed=TRUE),], aes(x=k, y=rank_sum, color=config))+
  geom_line()+
  scale_y_continuous(name='Normed Rank Sum over all CVIs')+
  scale_x_continuous(name='# Clusters (k)')+
  scale_color_discrete('Algorithm')+
  ggtitle('Rank Sum of best algorithmic configuration within dataset variations (y) and gamma (x) over k with n=1')+
  facet_grid(cols=vars(gamma_distance), rows=vars(labs))+
  theme(strip.text.y.right = element_text(angle = 0))

p

ggsave(filename = '../figures/grid_search/rank_sum_within_configuration_n=1.pdf', p, height = 30, width = 20)




#plot for mean(n)

mean_rank <- sub %>% group_by(pick_success, k,sigma,eta, windowsize) %>% transmute(avg_rank_sum = median(rank_sum),
                                                                                pick_success = pick_success,
                                                                                k=k,sigma=sigma,eta=eta,
                                                                                windowsize=windowsize,
                                                                                config=config,
                                                                                gamma_distance=gamma_distance,
                                                                                grps = paste('sigma:', sigma,
                                                                                             '_eta:', eta, '_window:',
                                                                                             windowsize, sep=''))

p <- ggplot(mean_rank, aes(x=k, y=avg_rank_sum, color=config))+
  geom_line()+
  scale_y_continuous(name='Median Normed Rank Sum over all CVIs')+
  scale_x_continuous(name='# Clusters (k)')+
  scale_color_discrete('Algorithm')+
  ggtitle('Rank Sum of best algorithmic configuration within dataset variations (y) and gamma (x) over k with n=1')+
  facet_grid(cols=vars(gamma_distance), rows=vars(grps))+
  theme(strip.text.y.right = element_text(angle = 0))

p

ggsave(filename = '../figures/grid_search/rank_sum_within_configuration_meadian(n).pdf', p, height = 30, width = 20)



##distributions over gammas in best picks
ident <- c()
for(a in algos){
  ident[length(ident)+1] <- paste("gamma_all_", a, "_Sil-D-COP-DB-DBstar-CH-SF", sep='')

}

sub <- df_reeval[df_reeval$pick_success %in% ident,]
sub$method <- ifelse(is.na(sub$method), 'partitional', paste('hiera.(', sub$method, ')', sep=''))
gamma.labs <- ifelse(gammas == 0, 'dtw_basic', paste('sdtw|gamma=', gammas, sep=''))
names(gamma.labs) <- as.character(gammas)

p <- ggplot(sub, aes(x=k))+
  geom_bar(stat='count')+
  #  geom_vline(data=cdat, aes(xintercept=k.mean,  colour=method),
  #             linetype="dashed", size=1)+
  facet_grid(rows = vars(method),
             cols=vars(gamma_distance),
             labeller = labeller(gamma_distance = gamma.labs))+
  
  ggtitle('Count over best picks (given only given algorithmic config was used and that k is given)')+
  theme(strip.text.y.right = element_text(angle = 0))

p

ggsave(filename = '../figures/grid_search/reevaluation_gamma_for_k.pdf', height = 10, width = 15)

