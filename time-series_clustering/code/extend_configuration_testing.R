#@Author: Marcel. H. Schubert
#!!!Note!!!
#!!!run only if all results on different hierarchical levels are needed!!!!

#script partially reruns simulation exercise from before and fills in blanks; i.e. only the best configuration per group was run and saved. Now, the script runs best configuration for given gammas and or given algorithm
# as well as other hierarchical levels of such kind -> number of results saved increases tremendously. Only needed for in-depth analysis

gen_filename_list <- function(path_to_dir){
  
  print('load the file names...')
   
  comp_list <- list.files(path=path_to_dir, pattern = 'comp_[a-z0-9]*\\.rds', recursive = FALSE)
  
  return(comp_list)
}

make_initial_grid <- function(grpsize=4, numrounds=10, endowment=20, n=1:6,
                              sigma=seq(0.6, 0.9, 0.1), eta=seq(0.6, 0.9, 0.1),
                              k=c(5, 8, seq(10,40, 5)), windowsize=1:3){
  source('utils/helpers.R')
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
  
  return(grid)
}

add_sha_info <- function(grid){
  require(openssl)
  
  grid <- apply(grid, MARGIN=1, function(row){

    idt <- sha1(x=paste('all_compositions', 'multivariate', row['sigma'], row['eta'], row['grpsize'],
                        row['numrounds'], row['endowment'],
                         row['windowsize'], row['n'], row['k'], TRUE, sep=""))
    row['idt'] <- idt

    return(as.data.frame(t(as.matrix(row))))
    
  })
  grid <- do.call(rbind, grid)
  return(grid)
  
}

add_information <- function(df_eval, config, TSobject, data, gridinfo){
  require(openssl)
  require(utils)
  library(strex)
  source('utils/evaluation_functions.R')
  
  print('extract variables...')
  algo <- TSobject@type
  
  cols <- colnames(config)
  mask <- cols %in% colnames(df_eval) 
  
  #write config information into df
  df_eval[1, cols[mask]] <- config[, cols[mask]]
  #print('saved internals in dataframe...')

  #make labels in same order as datalist
  labels <- lapply(names(TSobject@datalist), function(x, dat){
    type <- unique(dat[dat$uid == as.numeric(x), 'type'])
    data.frame(type=type,
               uid = as.numeric(x))
    
  }, dat = data[, c('uid', 'type')])
  labels <- do.call(rbind, labels)
  
  
  res <- try(eval_result(comparison = TSobject , labels$type, recall_weight = 0.5,tsobject = TRUE))
  #df with true type colum and cluster column
  df_clus <- res[[1]]
  #df with type colum and cluster column as well as frequency of type within cluster; holds type-wise val results
  df_clus1 <- res[[2]]
  #df providing a mapping betweent type and cluster; holds cluster-wise eval metrics results
  df_clus2 <- res[[3]]
  matching_dic <- res[[4]]
  #print(paste('made eval dic', sep=''))
  
  
  summa <- df_clus1 %>% group_by(type) %>% slice_max(freq, .preserve=TRUE) %>% summarise(max_freq = max(freq))
  
  mean_freq <- mean(summa$max_freq)
  
  med_freq <- median(summa$max_freq)
  
  
  ratio_c_t<-length(unique(df_clus2$type))/length(unique(types))
  
  # append to table
  df_clus$sha1_ident <- gridinfo$idt
  df_eval[1,"grp_comp"]<- "grp: all"
  df_eval[1,"types"]<- paste(sort(unique(labels$type)), collapse = '')
  df_eval[1,"measure"]<- 'multivariate'
  df_eval[1,"ratio"]<- ratio_c_t
  df_eval[1,'avg_max_share'] <- mean_freq
  df_eval[1,'median_share'] <- med_freq
  df_eval[1,"config"]<- algo
  df_eval[1,"sigma"]<- as.numeric(as.character(gridinfo$sigma))
  df_eval[1,"eta"]<- as.numeric(as.character(gridinfo$eta))
  df_eval[1,"grpsize"]<- as.numeric(as.character(gridinfo$grpsize))
  df_eval[1,"numrounds"]<- as.numeric(as.character(gridinfo$numrounds))
  df_eval[1,"endowment"]<- as.numeric(as.character(gridinfo$endowment))
  df_eval[1,"windowsize"]<- as.numeric(as.character(gridinfo$windowsize))
  df_eval[1,"n"]<- as.numeric(as.character(gridinfo$n))
  df_eval[1,"identifier"] <- as.character(gridinfo$idt)
  
  df_clus1$identifier <- gridinfo$idt
  
  df_eval <- add_df_clus_info(df_eval, df_clus1)
  df_eval <- write_metrics(df_eval, res[[3]], res[[2]], res[[4]], colnames(df_eval), types, length(unique(df_clus2$clus)))
  
  
  print('added all information to df...')
  
  return(df_eval)
}

ranking <- function(comp_org, data, cvis, algos, gridinfo, gamma = 'all'){
  
  source('utils/init_df_eval.R')
  
  if(gamma != 'all' & !is.na(gamma)){
    comp_org[["results"]] <- lapply(comp_org[["results"]], function(x, gamma){
      
      #x[x$gamma_distance %in% c(NA, gamma),]
      x[x$gamma_distance %in% c(gamma),]

      
    }, gamma=gamma)
  }
  
  #iterate over algos
  #print(gamma)
  for(i in 1:length(algos)){
    algo <- algos[i]
    #print(algo)
    #algo + all cvis
    ret_list <- internal_ranking(comp_org, cvis, algos = algo,
                                 add_rank = TRUE, add_normalized = FALSE,
                                 add_inverted = FALSE, config_object_only = TRUE)
    
    config_sel <- ret_list[[1]]
    TSobject <- ret_list[[2]]
    rm(ret_list)
    
    tmp <- add_information(empty_frame, config_sel, TSobject, data, gridinfo)
    tmp[1,"pick_success"] <- paste('gamma', gamma, algo, paste(cvis, collapse='-'), sep='_')
    if(i ==1){
      df_eval <- tmp
    }else{
      df_eval$pick_success <- as.character(df_eval$pick_success)
      df_eval <- dplyr::bind_rows(df_eval, tmp)
    }
    rm(tmp)
    
    #if we are in a gamma run we need to do the normal pick for that specific gamma also
    if(gamma != 'all' & !is.na(gamma)){
      ret_list <- internal_ranking(comp_org, cvis, algos = algos,
                                   add_rank = TRUE, add_normalized = FALSE,
                                   add_inverted = FALSE, config_object_only = TRUE)
      
      config_sel <- ret_list[[1]]
      TSobject <- ret_list[[2]]
      rm(ret_list)
      tmp <- add_information(empty_frame, config_sel, TSobject, data, gridinfo)
      tmp[1,"pick_success"] <- paste('gamma', gamma,paste(algos, collapse='-'), paste(cvis, collapse='-'), sep='_')
      df_eval$pick_success <- as.character(df_eval$pick_success)
      df_eval <- dplyr::bind_rows(df_eval, tmp)
    }
    
    
    
    
    #iterate over cvis
    for(cvi in cvis){
      #print(cvi)
      
      if(i == 1){
        #all algos # cvi
        #print(i)
        ret_list <- internal_ranking(comp_org, cvi, algos = algos,
                                     add_rank = TRUE, add_normalized = FALSE,
                                     add_inverted = FALSE, config_object_only = TRUE)
        
        config_sel <- ret_list[[1]]
        TSobject <- ret_list[[2]]
        rm(ret_list)
        
        tmp <- add_information(empty_frame, config_sel, TSobject, data, gridinfo)
        tmp[1,"pick_success"] <- paste('gamma', gamma, paste(algos, collapse='-'), cvi, sep='_')
        df_eval$pick_success <- as.character(df_eval$pick_success)
        df_eval <- dplyr::bind_rows(df_eval, tmp)
        rm(tmp)
      }
      #algo # cvi
      ret_list <- internal_ranking(comp_org, cvi, algos = algo,
                                   add_rank = TRUE, add_normalized = FALSE,
                                   add_inverted = FALSE, config_object_only = TRUE)
      
      config_sel <- ret_list[[1]]
      TSobject <- ret_list[[2]]
      rm(ret_list)
      
      tmp <- add_information(empty_frame, config_sel, TSobject, data, gridinfo)
      tmp[1,"pick_success"] <- paste('gamma', gamma, algo, cvi, sep='_')
      df_eval$pick_success <- as.character(df_eval$pick_success)
      df_eval <- dplyr::bind_rows(df_eval, tmp)
      
      rm(tmp)       
      
      #end cvi iteration
    }
    
    
    #end algo iteration 
  }
  
  return(df_eval)
}

make_outname <- function(savepath, grid, types, task_index, num_tasks){
  
  # outname_part<-  paste( "fnd_confg_variable_grid", 
  #                        "multivar", TRUE,
  #                        'internal', TRUE,
  #                        'k', paste(c(min(grid$k),max(grid$k)), collapse = "-"),
  #                        "grp_types", paste(types, collapse = ''),
  #                        "grpsize", paste(c(min(grid$grpsize),max(grid$grpsize)), collapse = "-"),
  #                        "numrounds",paste(c(min(grid$numrounds),max(grid$numrounds)), collapse = "-"),
  #                        "endowment", paste(c(min(grid$endowment),max(grid$endowment)), collapse = "-"),
  #                        "n", paste(c(min(grid$n),max(grid$n)), collapse="-"),
  #                        "sigma", paste(c(min(grid$sigma),max(grid$sigma)), collapse = "-"),
  #                        "eta", paste(c(min(grid$eta),max(grid$eta)), collapse = "-"),
  #                        "window", paste(c(min(grid$windowsize),max(grid$windowsize)), collapse = "-"),
  #                        'ind_groups',FALSE,
  #                        sep='_')
  
  outname <- paste(savepath, '/part_',task_index, '_', num_tasks ,'_df_eval_reevaluate_', '.rds', sep='')
  
  return(outname)
}

reevaluate <- function(cvis, algos, loadpath, savepath,
                       grpsize=4, numrounds=10, endowment=20, n=1:6,
                       sigma=seq(0.6, 0.9, 0.1), eta=seq(0.6, 0.9, 0.1),
                       k=c(5, 8, seq(10,40, 5)), windowsize=1:3, task_index, num_tasks){
  source('utils/helpers.R')
  
  #make blank df
  
  df_eval <- data.frame(pick_success = character())
  
  if(cvis == 'all'){
    cvis <- c('Sil', 'D', 'COP', 'DB', 'DBstar', 'CH', 'SF')
  }

    
  comp_list <- gen_filename_list(loadpath)
  #ordering for jobs
  comp_list <- sort(comp_list)
  
  parts <- partial_grid(comp_list, c(num_tasks, task_index))
  

  if(parts[1] > parts[2]){
    print("nothing left to do for me...quit")
    quit(save = "no", status = 0, runLast = FALSE)  
  }
  comp_list <- comp_list[parts[1]:parts[2]]
  
  
  grid <- make_initial_grid(grpsize=grpsize, numrounds=numrounds, endowment=endowment, n=n,
                            sigma=sigma, eta=eta,
                            k=k, windowsize=windowsize)
  
  grid <- add_sha_info(grid)
  
 #iterate over comparison objects
  types <- c()
  for(file in comp_list){
    #load data
    idt <- strsplit(strsplit(file, '_', fixed = TRUE)[[1]][2], '.', fixed = TRUE)[[1]][1]

    #grid info
    gridinfo <- grid[grid$idt == idt, ]
    print(gridinfo$sigma)
    
    comp_org <- try(readRDS(file= paste(loadpath, file, sep='/')))
    if(class(comp_org) == 'try-error'){
      print(paste(task_index, 'failed to open', file, sep=' '))
      
      next
    }
    df <- readRDS(file=paste(loadpath, '/..',"/df_clus_", idt, '.rds', sep =""))
    data <- readRDS(file=paste(loadpath, '/..',"/df_data_", idt, '.rds', sep =""))
    
    #add types
    types <- c(types, unique(data$type))
    
    comp_org[['pick']] <- NULL
    
    if(algos == 'all'){
      
      algos_do <- names(comp_org[['results']])
    }else{
      algos_do <- algos
    }
    
    get_results(comp_org[['results']], algos_do, gridinfo, 
                file.path(paste(loadpath, '/../reevaluation/comparison_data_', idt, '.rds', sep='')))
    
    tmp <- ranking(comp_org, data, cvis, algos_do, gridinfo, gamma = 'all')
    
    gammas <- unique(c(sapply(comp_org[["results"]], function(x){unique(x$gamma_distance)})))
    for(gamma in gammas){
      if(is.na(gamma)){
        next
      }
      tmp <- dplyr::bind_rows(tmp, ranking(comp_org, data, cvis, algos_do, gridinfo, gamma = gamma))
    }
    
    saveRDS(tmp, file = file.path(paste(loadpath, '/../reevaluation/df_reeval_', idt, '.rds', sep='')))
    print(paste('done with file', file),sep =' ')
    rm(comp_org)
    
    ##concat df
    df_eval <- dplyr::bind_rows(df_eval, tmp)
   #end file iteration 
  }
  
  types <- sort(unique(types))
  outname <- make_outname(savepath, grid, types, task_index, num_tasks)
  saveRDS(df_eval, file = outname)
  

  return(df_eval)
}

task_index <- as.numeric(Sys.getenv("SLURM_PROCID"))+1
num_tasks <- as.numeric(Sys.getenv('SLURM_STEP_TASKS_PER_NODE'))

if(is.na(task_index) |is.null(task_index)){
  task_index <-1
  num_tasks <- 1
}

types <- c(1,2,3,4,5)
savepath = '../Data/simulation/reevaluation'
loadpath = '../Data/simulation/workers_tmp/multivariate/comparison_objects'

df_eval <- reevaluate(cvis = 'all', algos='all', loadpath = loadpath, savepath = savepath,
                      grpsize=4, numrounds=10, endowment=20, n=1:6,
                      sigma=seq(0.6, 0.9, 0.1), eta=seq(0.6, 0.9, 0.1),
                      k=c(5, 8, seq(10,40, 5),  seq(42, 54, 2)), windowsize=1:3, task_index, num_tasks)
