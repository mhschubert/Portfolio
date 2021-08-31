#@author Marcel H. Schubert

find_cnfg_grp_ <-function(types, grpsize, numrounds, endowment, n, sigma, eta, windowsize, k, seed=42,
                          add_errors, rand_first, random_subset, grp_comp_ls = NA, measures= c('contribution', 'ratio', 'diff'),
                          ind_groups= FALSE, testrun=FALSE, multivariate, savedf, empty_frame, file ='_basic',
                          path="", inner_parallel = FALSE, worker=1, part_grid = NA, filter_static = FALSE){

  
  require(dtw)
  require(dtwclust)
  require(tidyr) 
  require(dplyr)
  require(stringr)

  require(arrangements)
  require(doParallel)
  require(openssl)
  require(utils)
  library(strex)
 
  
  source('utils/cnfg_lst.R')
  source('utils/data_simulation.R')
  source('utils/evaluation_functions.R')
  source('utils/helpers.R')
  
  
  ##create identifier for whether we do a single group or all groups
  if(is.na(grp_comp_ls)){
    grp <- 'all_compositions'
  }
  else{
    grp <- paste(grp_comp_ls[[1]] , collapse = '')
  }
  
  #check whether the current data run was already precomputed - no need to redo it
  #calc unique hash identifier
  idts <- sha1(x=paste(grp, sigma, eta, grpsize, numrounds, endowment, windowsize, n, k, multivariate, sep=""))
  
  fn <- paste(file,'_', idts, '.txt', sep = '')

  #simulate data
  
  simdat_grp <- generate_data(types, grpsize, numrounds, endowment, rand_first, n, random_subset, 
                              add_errors, sigma, eta, seed, grp_comp_ls, normalize = TRUE)
  
  if(filter_static){
    simdat_grp <- simdat_grp[simdat_grp$type != 1 & simdat_grp$type != 4]
    
  }
  lab <- unique(simdat_grp$type)
  #make unique idea for ls-function
  simdat_grp$uid<-as.numeric(paste(simdat_grp$uid, as.numeric(gsub(" ", "",as.character(simdat_grp$grp_cmpstn), fixed=TRUE)), sep=""))
  
  dat <- simdat_grp
  print(paste(worker, ': made data', sep=''))
  
  

  num_clust <- k

  if(windowsize == "NULL"){
    windowsize <- eval(parse(text= windowsize))
  }
  if(multivariate){
    ##set configs for multivariate
    print('in define configs, multivariate')
    measures <- c('multivariate')
    print('set measures')
    clust_typs <- c("p","h" )
    cfgs <- define_config_multi(num_clust, windowsize, clust_typs)
    
  }else{
    print('in define configs, univariate')
    cfgs <- define_config_uni(num_clust, windowsize)
    clust_typs <- c("p","h", "t","f" )
    #clust_typs <- c("h")
    
  }
  
  #adapt configs according to table 2 in Sarda-Espinosa 2019
  cfgs <- adapt_configs(cfgs)
  
  
  
  print(paste(worker, ': defined configs', sep=''))
  
  
  
  
  #######
  #clust#
  #######
  
  for (m in 1:length(measures)) {
    
    ##set entry in dataframe
    if(random_subset == "specific"){
      resstr <- paste("grp: ", paste(grp_comp_ls[[1]], collapse = ""))
      
    }
    else{
      resstr <- "grp: all"
      
    }
    
    #calc unique hash identifier
    idt <- sha1(x=paste(grp, measures[m], sigma, eta, grpsize, numrounds, endowment, windowsize, n, k, multivariate, sep="" ))

    print(paste(worker, idt, ':', measures[m], sep=' '))
    
    
    #set an empty frame to save to
    df_eval_new <- empty_frame
    
    #m=1  #for testing
    # generate input list
    ls <- try(make_input_list(simdat_grp, multivariate, measures[m]))
    if(class(ls) == class('try_error')){
      print('Skipped because datalist could not be generated')
      print(paste(resstr, ", input: ", measures[m], 
                  " grpsize: ", grpsize, " numrounds: ", numrounds,
                  ", endowment: ", endowment, ", sigma: ", sigma,
                  ", eta: ", eta, ", windowsize: ", windowsize, ", n: ", n,sep="" ))
      df_eval_new[1, ] <- NA
      df_eval_new[1,"grp_comp"]<- resstr
      df_eval_new[1,"types"]<- paste(lab, collapse = '')
      df_eval_new[1,"measure"]<- measures[m]
      df_eval_new[1,"sigma"]<- sigma
      df_eval_new[1,"eta"]<- eta
      df_eval_new[1,"grpsize"]<- grpsize
      df_eval_new[1,"numrounds"]<- numrounds
      df_eval_new[1,"endowment"]<- endowment
      df_eval_new[1,"windowsize"]<- windowsize
      df_eval_new[1,"n"]<- n
      df_eval_new[1,"identifier"] <- rep(paste(idt, sep=""), dim(df_eval_new)[1])
      df_eval_new[1, is.na(df_eval_new[1,])] <- 'try_error'
      print('wrote try-errors')
      savedf<-rbind(savedf, df_eval_new) # contains all iformation after all iterations
      #skip rest of iteration since calculation did not work
      next
      
    }

    labels<- generate_labels(simdat_grp)
    
    
    if(testrun){
      ls <- ls[1:70]
      labels <- labels[1:70]
    }
    

    
    # Using all external CVIs and majority vote
    

    if(!('f' %in% clust_typs)){
      internal_evaluators <- cvi_evaluators("internal", fuzzy=FALSE)  
    }else{
      internal_evaluators <- cvi_evaluators("internal", fuzzy=TRUE)
    }
    
    score_internal <- internal_evaluators$score
    
    pick_majority_int <- internal_evaluators$pick
    
    # Number of configurations
    num_configs <- 0
    for(nm in names(cfgs)){
      num_configs <- num_configs + length(cfgs[nm][[1]])
    }
    
    cat("\nTotal number of configurations without considering optimizations:",
        sum(num_configs),
        "\n\n")
    print(paste(grp, sigma, eta, grpsize, numrounds, endowment, windowsize, n, k, sep=" "))
    print(paste('n is ', n, sep=''))
    print(paste("length of data is ", length(ls), sep=''))
    #cluster
    ##reinterpolate to same length as before to get numeric recognition
    print(paste('multivariate is ', multivariate, sep=''))
    if(multivariate){
      #nothing happens here except making the data numeric which for some reason it is not;
      #length numrounds-1 or length(ls[[1]][,1]) as first round is excluded
      ls <- reinterpolate(ls, length(ls[[1]][,1]), multivariate = TRUE)
    }
    print('start comparisons...')
    
    comparison <- NULL
    comparison <- try(compare_clusterings(ls, 
                                          types = clust_typs, 
                                          configs = cfgs,
                                          seed = seed, 
                                          trace = FALSE,
                                          score.clus = score_internal,
                                          #pick.clus = pick_majority_int,
                                          return.objects = TRUE))
    
    if((!is.null(comparison)) & (class(comparison) != "try-error")){
      saveRDS(comparison, file =paste(path, '/comparison_objects', '/comp_',idt , ".rds", sep=""))
      
    }
    #set pick success to true and change it below in case it failed
    if( (!is.null(comparison)) & (class(comparison) != "try-error") ){
      
      #df_eval_new[1, 'pick_success'] <- TRUE
      
      #if(is.null(comparison[["pick"]])){
      #  print('do a custom pick...')
      #  comparison <- internal_picker(comparison)
      comparison <- internal_ranking(comparison)
      #  df_eval_new[1, 'pick_success'] <- FALSE
      #}
    }
    
  
    if(is.null(comparison[["pick"]]) |is.null(comparison) | class(comparison) == "try-error"){
      print(paste('Skipped iteration because try-error ==', class(comparison) == "try-error",
                  'and comparison-object is NULL ==', class(comparison) == class(NULL),
                  'and pick object is ==', class(comparison[["pick"]][["object"]]),
                  sep=' '))
      print(paste(resstr, ", input: ", measures[m], 
                  " grpsize: ", grpsize, " numrounds: ", numrounds,
                  ", endowment: ", endowment, ", sigma: ", sigma,
                  ", eta: ", eta, ", windowsize: ", windowsize, ", n: ", n,sep="" ))
      #df_eval_new[1, ] <- NA
      df_eval_new[1,"grp_comp"]<- resstr
      df_eval_new[1,"types"]<- paste(unique(labels), collapse = '')
      df_eval_new[1,"measure"]<- measures[m]
      df_eval_new[1,"sigma"]<- sigma
      df_eval_new[1,"eta"]<- eta
      df_eval_new[1,"grpsize"]<- grpsize
      df_eval_new[1,"numrounds"]<- numrounds
      df_eval_new[1,"endowment"]<- endowment
      df_eval_new[1,"windowsize"]<- windowsize
      df_eval_new[1,"n"]<- n
      df_eval_new[1,"identifier"] <- rep(paste(idt, sep=""), dim(df_eval_new)[1])
      df_eval_new[1, is.na(df_eval_new[1,])] <- 'try_error'
      if(class(comparison[["pick"]][["object"]]) == class(NULL)){
        df_eval_new[1,"config"]<- 'try-error'
      }
      
      print('wrote try-errors')
      savedf<-rbind(savedf, df_eval_new) # contains all iformation after all iterations
      #skip rest of iteration since calculation did not work
      next
      
    }
    
    print(paste(worker, ': done comparisons with internals', sep=''))
    #console output
    
    print('extract variables...')
    #algo <- tolower(strsplit(class(comparison[["pick"]][["object"]]), split = "TSClusters", fixed = TRUE)[[1]])
    algo <- tolower(comparison[["pick"]][["object"]]@type)
    config <- comparison[["pick"]]$config
    
    print(paste('algorithm ', algo, ' is best...', sep=""))
    
    
    used_cnfg <- str_first_number(comparison$pick$config$config_id)
    #for some reason the numbers are not starting at 1 - we fix that here (numbers are still sequential)
    subtract <- min(str_first_number(names(comparison[[paste('objects', algo, sep='.')]])))-1
    
    used_cnfg <- used_cnfg - subtract
    
    cfg_all <-cfgs
    cfgs[[algo]] <- cfgs[[algo]][used_cnfg,]
    
    cfgs[[algo]]$method <- comparison$pick$config$method
    

    print('done extracting...')
    #save results in frame
    df_eval_new[1, colnames(comparison$pick$config)] <- comparison$pick$config[1,]
    
    print('saved internals in dataframe...')
    
    res <- try(eval_result(comparison, labels,recall_weight = 0.5))
    
    if(class(res) == "try-error"){
      print('Skipped iteration because evaulation is try-error:')
      print(paste(resstr, ", input: ", measures[m], 
                  " grpsize: ", grpsize, " numrounds: ", numrounds,
                  ", endowment: ", endowment, ", sigma: ", sigma,
                  ", eta: ", eta, ", windowsize: ", windowsize, ", n: ", n,sep="" ))
      #df_eval_new[1, ] <- NA
      df_eval_new[1,"grp_comp"]<- resstr
      df_eval_new[1,"types"]<- paste(unique(labels), collapse = '')
      df_eval_new[1,"measure"]<- measures[m]
      df_eval_new[1,"sigma"]<- sigma
      df_eval_new[1,"eta"]<- eta
      df_eval_new[1,"grpsize"]<- grpsize
      df_eval_new[1,"numrounds"]<- numrounds
      df_eval_new[1,"endowment"]<- endowment
      df_eval_new[1,"windowsize"]<- windowsize
      df_eval_new[1,"n"]<- n
      df_eval_new[1,"identifier"] <- idt
      df_eval_new[1,"config"]<- comparison[["pick"]][["object"]]@type
      df_eval_new[1, is.na(df_eval_new[1,])] <- 'try_error'
      savedf<-rbind(savedf, df_eval_new) # contains all iformation after all iterations
      #skip rest of iteration since calculation did not work
      next
      
    }   
    
    
    
    
    
    #df with true type colum and cluster column
    df_clus <- res[[1]]
    #df with type colum and cluster column as well as frequency of type within cluster; holds type-wise val results
    df_clus1 <- res[[2]]
    #df providing a mapping betweent type and cluster; holds cluster-wise eval metrics results
    df_clus2 <- res[[3]]
    matching_dic <- res[[4]]
    print(paste(worker, ': made eval dic', sep=''))
    
    
    summa <- df_clus1 %>% group_by(type) %>% slice_max(freq, .preserve=TRUE) %>% summarise(max_freq = max(freq))

    mean_freq <- mean(summa$max_freq)

    med_freq <- median(summa$max_freq)

    
    ratio_c_t<-length(unique(df_clus2$type))/length(unique(types)) #why is this here - I think I do not need it, probably a leftover
    
    # append to table
    df_clus$sha1_ident <- rep(paste(idt, sep=''), dim(df_clus)[1])
    df_eval_new[1,"grp_comp"]<- resstr
    df_eval_new[1,"types"]<- paste(unique(labels), collapse = '')
    df_eval_new[1,"measure"]<- measures[m]
    df_eval_new[1,"ratio"]<- ratio_c_t
    df_eval_new[1,'avg_max_share'] <- mean_freq
    df_eval_new[1,'median_share'] <- med_freq
    df_eval_new[1,"config"]<- comparison[["pick"]][["object"]]@type
    df_eval_new[1,"sigma"]<- sigma
    df_eval_new[1,"eta"]<- eta
    df_eval_new[1,"grpsize"]<- grpsize
    df_eval_new[1,"numrounds"]<- numrounds
    df_eval_new[1,"endowment"]<- endowment
    df_eval_new[1,"windowsize"]<- windowsize
    df_eval_new[1,"n"]<- n

  
    df_eval_new[1,"identifier"] <- rep(paste(idt, sep=""), dim(df_eval_new)[1])

    df_clus1$identifier <- rep(paste(idt, sep=""), dim(df_clus1)[1])
    

    df_eval_new <- add_df_clus_info(df_eval_new, df_clus1)
   
    

    ##delete object
    rm(comparison)
    #write evaluation metrics into data frame
    print(paste(worker, ': made eval', sep=''))
    df_eval_new <- write_metrics(df_eval_new, res[[3]], res[[2]], res[[4]], colnames(savedf), types, length(unique(df_clus2$clus)))
    print('wrote metrics...')
    df <- dplyr::bind_rows(lapply(seq_along(ls),
                                  function(i, l, nm ){
                                    data.frame(uid=rep(nm[i], length(l[[i]])), data=l[[i]], period = seq_along(l[[i]]), stringsAsFactors=FALSE)
                                  },
                                  l=ls,
                                  nm = names(ls)))
    
    df <- merge(df, df_clus, by='uid')
    
    df$sha1_ident <- paste(idt, sep='')
    
    df$measure <- measures[m]
    
    df$multivariate <- multivariate
    
    if(!testrun){
      saveRDS(df, file=paste(path, "/df_data_", idt, ".rds", sep =""))
      saveRDS(df_clus1, file=paste(path, '/df_clus_', idt, '.rds', sep=""))
    }else{
      print('would have saved to tmp-file...')
      print(paste(path, "/df_data_", idt, ".rds", sep =""))
    }

    
    savedf<-rbind(savedf, df_eval_new) # contains all information after all iterations
    
    
  }
  
  result <- savedf
  if(!testrun){
    saveRDS(savedf,file=paste(path, "/fin_df_eval_", sha1(x=paste(grp, sigma, eta, grpsize, numrounds, endowment, windowsize, n, k, 
                                                                  multivariate, sep="")), ".rds", sep=""))
    print(paste(worker, ': saved evaluations in workerfiles', sep=''))
  }else{
    print('would have saved to...')
    print(paste(path, "/fin_df_eval_", sha1(x=paste(grp, sigma, eta, grpsize, numrounds, endowment, windowsize, n, k, 
                                                    multivariate, sep="")), ".rds", sep=""))
    returner <<- savedf
  }
  return(savedf)
}


delete_computed_rows <- function(grid,multivariate, path, worker, grp_comp =NA){
  require(openssl)
  initial <- dim(grid)[1]
  removal_vector <- c()
  for(i in 1:(dim(grid)[1])){
    sigma <- grid$sigma[i]
    eta <- grid$eta[i]
    grpsize <- grid$grpsize[i]
    numrounds <- grid$numrounds[i]
    endowment <- grid$endowment[i]
    windowsize <- grid$windowsize[i]
    n <- grid$n[i]
    k <- grid$k[i]
    
    ##create identifier for whether we do a single group or all groups
    if(is.na(grp_comp)){
      grp <- 'all_compositions'
    }
    else{
      grp <- paste(grp_comp , collapse = '')
    }
    
    #check whether the current data run was already precomputed - no need to redo it
    #calc unique hash identifier
    idts <- sha1(x=paste(grp, sigma, eta, grpsize, numrounds, endowment, windowsize, n, k, multivariate, sep=""))

    fn <- paste(file,'_', idts, '.txt', sep = '')
    
    #load file if it exists
    if(!testrun){
      if(file.exists(paste(path, "/fin_df_eval_", idts, ".rds", sep =""))){
        
        prec <- try(readRDS(paste(path, "/fin_df_eval_", idts, ".rds", sep ="")))
        ##check if reading successful
        if(class(prec) != "try-error" & class(prec) == 'data.frame'){
          #check if calculations were successful
          if(prec$config != 'try-error'){
            removal_vector[length(removal_vector)+1] <-i
          }
        }
      }
    }
  }
  print(paste('removal vector is', length(removal_vector), sep=' '))
  if(!testrun & class(c()) != class(removal_vector)){
    grid <- grid[-removal_vector,]
  }
  print(sprintf('remove %d computed rows from grid', length(removal_vector)))
  print(sprintf('%d rows are remaining to compute', initial - length(removal_vector)))
  
  if(initial - length(removal_vector) == 0){
    print("nothing left to do for me...quit")
    quit(save = "no", status = 0, runLast = FALSE) 
  }

  return(grid)
}





source('utils/helpers.R')
require(here)
args = commandArgs(trailingOnly=TRUE)
types <- c(1,2,3,4,5)
source('utils/init_df_eval.R')
grpsize <- 4
#numrounds <- seq(10, 30, 10)
numrounds <- 10
endowment <-seq(20, 20, 10)
n <- 1:6
#n <- c(1,3,6)
sigma <- seq(0.6, 0.9, 0.1)

eta <- seq(0.6, 0.9, 0.1)

k <- c(5, 8, seq(10,40, 5),  seq(42, 54, 2))
#k <- seq(42, 54, 2)
seed <- 42

add_errors <- TRUE
rand_first <- FALSE
filter_static <- FALSE
random_subset <- FALSE
ind_groups <- FALSE
multivariate <- TRUE

windowsize <- 1:3
#window_width <- c(3)

testrun <- FALSE

file <- 'df_eval'
if(testrun){
  args <- c('TRUE', '16', '1')
  multivariate <- as.logical('TRUE')
  #task_index <- as.numeric(Sys.getenv("SLURM_PROCID"))+1
  #num_tasks <- as.numeric(Sys.getenv('SLURM_STEP_TASKS_PER_NODE'))
  task_index <- 40
  num_tasks <- 40
}else{
  multivariate <- as.logical(args[1])
  task_index <- as.numeric(Sys.getenv("SLURM_PROCID"))+1
  num_tasks <- as.numeric(Sys.getenv('SLURM_STEP_TASKS_PER_NODE'))
}




#set working directory
wd <- getwd()
pd <- here()
if(!('r' %in% strsplit(wd, split ="/", fixed = TRUE)[[1]]) | ('r' %in% strsplit(pd, split ="/", fixed = TRUE)[[1]])){
  #check wether parent directory is at least found
  splt <- strsplit(pd, split ="/", fixed = TRUE)[[1]]
  stopifnot('time-series_clustering' %in% splt)
  #make path to script
  for(i in length(splt):1){
    if(splt[i] == "ehs"){
      break
      
    }
    else{
      end <- i-1
      splt <- splt[1:end]
    }
    
  }
  pd <- paste(splt, collapse = "/")
  wd <- paste(pd, "r", sep= "/")
  setwd(wd)
}else{
  splt <- strsplit(wd, split ="/", fixed = TRUE)[[1]]
  wd <- paste(splt[1:(which('r' == splt)[1])], collapse='/')
  setwd(wd)
}


#make filepaths depending on multivariate
if(multivariate){
  vari <- 'multivariate'
}else{
  vari <- 'univariate'
}
savepath <- paste(pd, "Data", "simulation", sep = "/")
workerpath <- paste(savepath, "workers_tmp", vari, sep = "/")
if(dir.exists(savepath)){
  dir.create(savepath, recursive = TRUE)
}
if(dir.exists(workerpath)){
  dir.create(workerpath, recursive = TRUE)
}
outname_part<-  paste( "fnd_confg_variable_grid", 
                       "multivar", multivariate,
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

workfile <- paste(workerpath, paste(file, 'worker', sep="_"), sep ='/')



vectorlist <- list(
  grpsize = grpsize,
  numrounds = numrounds,
  endowment = endowment,
  n = n,
  sigma = sigma,
  eta = eta,
  windowsize = windowsize,
  k = k
)

grid <- make_grid(vectorlist)




multivariate <- as.logical(args[1])
vec <- c(as.numeric(args[2]), as.numeric(args[3]))
len <- floor(dim(grid)[1]/num_tasks/vec[1])
  
  
##make partial grid for jobs
gridpart <- partial_grid(grid,vec)
print(paste('worker', task_index, 'has the parts', gridpart[1], gridpart[2], sep=' '))
grid <- grid[gridpart[1]:gridpart[2],]
  
grid <- delete_computed_rows(grid,multivariate, path=workerpath, worker=task_index ,grp_comp =NA)
  
#make aprtial grid for process tasks
gridpart <- partial_grid(grid, c(num_tasks, task_index))
grid <- grid[gridpart[1]:gridpart[2],]

  

if(testrun){
  grid <- grid[6,]
}


if((dim(grid)[1]) >= 1){

  print(sprintf('the remaining rows are %d', dim(grid)[1]))
  for(i in 1:(dim(grid)[1])){
    df_ret <- find_cnfg_grp_(types =types, grpsize = grid$grpsize[i], grid$numrounds[i],
                                   grid$endowment[i], grid$n[i], grid$sigma[i],
                                   grid$eta[i], grid$windowsize[i], grid$k[i],
                                   seed=42,add_errors=add_errors, rand_first=rand_first,
                                   random_subset=random_subset,grp_comp_ls = NA,
                                   measures= c('contribution', 'ratio', 'diff'),
                                   ind_groups= ind_groups, testrun=testrun, multivariate, savedf=df_eval,
                                   empty_frame=empty_frame, file = paste(workfile,task_index,i, sep='_'),
                                   path= workerpath,
                                   inner_parallel = inner_parallel,
                                   worker=(len*(task_index-1))+i, part_grid = gridpart, filter_static = FALSE)
  }
}else{
  print('everything is alreaydy done')
}

  
print('done')





