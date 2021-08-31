#@author Marcel H. Schubert

##############################################
#Script to concatenate results of dtw_configuration_testing.R if distributed across more than one node
##############################################


make_grid <- function(vectorlist){
  require(purrr)
  
  grid <- vectorlist %>% cross_df()
  
  return(grid)
  
}




load_concat <- function(types, grpsize, numrounds, endowment, n, sigma, eta, windowsize, k = 0, seed=42,
                        add_errors, rand_first, random_subset, ind_groups= FALSE, multivariate, inner_parallel =FALSE,
                        testrun=FALSE, file='df_eval'){
  
  
  
  require(dtw)
  require(dtwclust)
  require(tidyr) 
  require(dplyr)
  require(stringr)
  require(ggplot2)
  require(grid)
  require(gridExtra)
  require(arrangements)
  require(doParallel)
  require(here)
  require(openssl)
  
  
  
  #set working directory
  #set working directory
  wd <- getwd()
  pd <- here()
  if(!('code' %in% strsplit(wd, split ="/", fixed = TRUE)[[1]]) | ('code' %in% strsplit(pd, split ="/", fixed = TRUE)[[1]])){
    #check wether parent directory is at least found
    splt <- strsplit(pd, split ="/", fixed = TRUE)[[1]]
    stopifnot('time-series_clustering' %in% splt)
    #make path to script
    for(i in length(splt):1){
      if(splt[i] == "time-series_clustering"){
        break
        
      }
      else{
        end <- i-1
        splt <- splt[1:end]
      }
      
    }
    pd <- paste(splt, collapse = "/")
    wd <- paste(pd, "code", sep= "/")
    setwd(wd)
  }else{
    splt <- strsplit(wd, split ="/", fixed = TRUE)[[1]]
    wd <- paste(splt[1:(which('code' == splt)[1])], collapse='/')
    setwd(wd)
  }
  
  
  #make filepaths depending on multivariate
  if(multivariate){
    vari <- 'multivariate'
  }else{
    vari <- 'univariate'
  }
  savepath <- paste(pd, "Data/simulation", "results", sep = "/")
  workerpath <- paste(savepath, "workers_tmp", vari, sep = "/")
  
  if(k[1] != 0){
    internal <- TRUE
    k <- paste(c(min(k),max(k)), collapse = "-")
    
  }else{
    internal <- FALSE
    k <- 'fixed'
  }
  
  
  outname_part<-  paste( "fnd_confg_variable_grid", 
                         "multivar", multivariate,
                         'internal', internal,
                         'k', k,
                         "grp_types", paste(types, collapse = ''),
                         "grpsize", paste(c(min(grpsize),max(grpsize)), collapse = "-"),
                         "numrounds",paste(c(min(numrounds),max(numrounds)), collapse = "-"),
                         "endowment", paste(c(min(endowment),max(endowment)), collapse = "-"),
                         "n", paste(c(min(n),max(n)), collapse="-"),
                         "sigma", paste(c(min(sigma),max(sigma)), collapse = "-"),
                         "eta", paste(c(min(eta),max(eta)), collapse = "-"),
                         "window", paste(c(min(windowsize),max(windowsize)), collapse = "-"),
                         'ind_groups',ind_groups,
                         sep='_')
  
  outname <- paste(file, outname_part, sep='_')
  outfile <- paste(paste(savepath, outname, sep="/"), '.rds', sep ='')
  
  
  
  
  
  print('load the file names...')
  
  df_eval_list <- list.files(path=workerpath, pattern = 'fin_df_eval_[a-z0-9]*\\.rds', recursive = FALSE)
  #df_data_list <- list.files(path=workerpath, pattern = 'df_data_[a-z0-9]*\\.rds', recursive = FALSE)
  df_reeval_list <- list.files(path=paste(workerpath, '../../reevaluation', sep='/'), pattern = 'part', recursive = FALSE)
  
  df_tot_list <- list.files(path=paste(workerpath, 'reevaluation', sep='/'),
                            pattern = 'comparison_data_[a-z0-9]*\\.rds', recursive = FALSE)
  
  
  df_eval <- readRDS(paste(workerpath, df_eval_list[1], sep="/"))
  
  df_reeval <- readRDS(paste(workerpath, '../../reevaluation', df_reeval_list[1], sep="/"))
  
  df_tot <- readRDS(paste(workerpath, 'reevaluation', df_tot_list[1], sep="/"))
  
  print('read in and concat files')
  print(paste('have', length(df_eval_list), 'files to load', sep=' '))
  for(i in 2:(length(df_eval_list))){
    file <- try(readRDS(paste(workerpath, df_eval_list[i], sep="/")))
    if(class(file) != 'try-error'){
      df_eval <- dplyr::bind_rows(df_eval, file)
    }
    else{
      print('try-error')
    }
    
  }
  
  
  print('save concat files to disk...')
  
  saveRDS(df_eval, file = outfile)
  
  print('read in and concat reeval parts')
  print(paste('have', length(df_reeval_list), 'files to load', sep=' '))
  for(i in 2:(length(df_reeval_list))){
    file <- try(readRDS(paste(workerpath, '../../reevaluation', df_reeval_list[i], sep="/")))
    
    if(class(file) != 'try-error'){
      df_reeval <- dplyr::bind_rows(df_reeval, file)
    }
    else{
      print('try-error')
    }
    
  }
  
  
  print('save reeval to disk...')
  
  saveRDS(df_reeval, file = paste(workerpath, '../..', 'reevaluation_gridsearch.rds', sep='/'))
  
  print('read in and concat result tables')
  print(paste('have', length(df_tot_list), 'files to load', sep=' '))
  
  for(i in 2:(length(df_tot_list))){
    file <- try(readRDS(paste(workerpath, 'reevaluation', df_tot_list[i], sep="/")))
    
    if(class(file) != 'try-error'){
      df_tot <- dplyr::bind_rows(df_tot, file)
    }
    else{
      print('try-error')
    }
    
  }
  
  
  print('save reeval to disk...')
  
  saveRDS(df_tot, file = paste(savepath, 'gridsearch_results.rds', sep='/'))
  write.csv(df_tot, file = paste(savepath, 'gridsearch_results.csv', sep='/'))
  
  print('done')
  return(df_eval)
  
}


types <- c(1,2,3,4,5)

grpsize <- 4
numrounds <- seq(10, 10, 10)
#numrounds <- 10
endowment <-seq(20, 20, 10)
n <- 1:6
sigma <- seq(0.6, 0.9, 0.1)
eta <- seq(0.6, 0.9, 0.1)
k <- c(5, 8, seq(10,40, 5),  seq(42, 54, 2))
seed <- 42
add_errors <- TRUE
rand_first <- FALSE
filter_static <- FALSE
random_subset <- FALSE
ind_groups <- FALSE
multivariate <- TRUE

windowsize <- 1:3


df_eval <- load_concat(types, grpsize, numrounds, endowment, n, sigma, eta, windowsize,k, seed=42,
                       add_errors=add_errors, rand_first=rand_first, random_subset=random_subset, multivariate = multivariate,
                       ind_groups=ind_groups, testrun=testrun)

