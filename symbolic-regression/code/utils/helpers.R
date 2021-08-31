#@author Marcel H. Schubert

remove_difficult <- function(diff_list, combinations){
  
  for(i in 1:dim(df)[1]){
    for(j in 1:length(diff_list)){
    }
  }
}

make_export <- function(){
  
  env <- globalenv()
  fun <- lapply(env, function(x){
    
    if(paste(class(x)) == 'function'){
      return(x)
    }
    else{
      return(NA)
    }
  })
  
  fun <- fun[!is.na(fun)]
  fun_names <- ls(globalenv())[ls(globalenv()) %in% names(fun)]
  return(fun_names)
}

make_grid <- function(vectorlist){
  require(purrr)
  
  grid <- vectorlist %>% cross_df()
  
  return(grid)
  
}

make_grid_list <- function(grpsize, numrounds, endowment, n, sigma, eta, windowsize){
  
  
  vectorlist <- list(
    grpsize = grpsize,
    numrounds = numrounds,
    endowment = endowment,
    n = n,
    sigma = sigma,
    eta = eta,
    windowsize = windowsize
  )
  
  return(vectorlist)
}

save_data <- function(outname, workerpath, outpath){
  
  filevec <- list.files(path = workerpath)
  boolVec <- grepl('df_data', filevec , fixed = TRUE)
  filevec <- filevec[boolVec]
  print(paste(workerpath, filevec[1], sep='/'))
  df <- readRDS(file = paste(workerpath, filevec[1], sep='/'))
  for(i in 2:length(filevec)){
    df <- rbind(df, readRDS(file = paste(workerpath, filevec[i], sep='/')))
  }
  
  saveRDS(df, paste(outpath, "/df_data_", outname, ".rds", sep=''))
  
  
  return(df)
}

centro_dist_pairs <- function(df, key){
  if(key$centroid == 'dba'){
    df <- df[df$distance == 'dtw_basic' | df$distance == 'sdtw',]
  }else if(key$centroid == 'sdtw_cent'){
    df <- df[df$distance == 'sdtw',]
  }else if(key$centroid == 'shape_extraction'){
    df <- df[df$distance == 'sbd',]
    
  }
  
  return(df)
}

adapt_configs <- function(cfgs){
  
  for(algo in names(cfgs)){
    if(algo != 'tadpole'){
      cfgs[algo][[1]] <- cfgs[algo][[1]] %>% group_by(centroid) %>%
        group_modify(~ centro_dist_pairs(df=.x, key=.y)) %>%
        bind_rows()
      cfgs[algo][[1]] <- ungroup(cfgs[algo][[1]])
    }
    
  }
  
  
  
  return(cfgs)
}


make_input_list <- function(simdat_grp, multivariate, measure){
  
  # generate input list
  id<-unique(simdat_grp$uid)
  ls <- list()
  
  if(multivariate){
    for (j in 1:length(id)) {  
      df_sb<-simdat_grp[simdat_grp$uid==id[j],]
      #remove first round as the lagged contribution does not exist here
      df_sb <- df_sb[df_sb$period != 1,]
      
      df_sb <- df_sb[order(df_sb$period),  c("contribution", "avg_others_l")]
      
      ls[[j]]<-df_sb
      #ls2 <<- ls
    }      
    
  }else{
    for (j in 1:length(id)) {  
      df_sb<-simdat_grp[simdat_grp$uid==id[j],]
      if(measure == 'diff' | measure == 'ratio'){
        #remove first period as that period has no meaning
        df_sb <- df_sb[df_sb$period != 1,]
        
      }
      df_sb <- df_sb[order(df_sb$period),  measure]
      ls[[j]]<-df_sb
      #ls3 <<- ls
    }
  }
  
  names(ls) <- id
  
  return(ls)
  
  
}

generate_labels <- function(simdat_grp){
  
  labels<-as.factor(unique(simdat_grp[,c('uid', 'type')]) %>% pull(type)) #dangerous because of ordering may not be static
  names(labels) <- unique(simdat_grp$uid)
  return(labels)
}

make_multivariate_numeric <- function(ls){
  
  ls <- lapply(ls, function(x){
    datavec <- c()
    for(i in 1:dim(x[[1]][2])){
      
      datavec <- c(datavec, as.vector(x[[1]][,i]))
      
    }
    
    ret <- matrix(data = datavec, nrow = dim(x[[1]][1]), ncol = dim(x[[1]][2]), byrow = FALSE)
    colnames(ret) <- colnames(x[[1]])
    return(ret)
  })
  
  return(ls)
}


add_df_clus_info <- function(df_eval, df_clus1){
  
  sizes <- df_clus1 %>% group_by(clus) %>% summarise(s = sum(n))
  for(i in 1:(dim(sizes)[1])){
    ##cluster_sizes
    df_eval[1, sprintf("size_cluster_%d", sizes$clus[i])] <- sizes$s[i]
    
  }
  
  for(i in 1:(dim(df_clus1)[1])){
    #cluster
    cluster_n <- sprintf("n_clust_%d_type_%d",unname(as.vector(df_clus1$clus[i])),  unname(as.vector(df_clus1$type[i])))
    #print(cluster_n)
    #freq
    cluster_freq <- sprintf("freq_clust_%d_type_%d",unname(as.vector(df_clus1$clus[i])),  unname(as.vector(df_clus1$type[i])))
    
    df_eval[1, cluster_n] <- df_clus1$n[i]
    df_eval[1, cluster_freq] <- df_clus1$freq[i]
  }
  return(df_eval)
}


fill_zeros <- function(df){
  #function to fill zeros into columns for size_clust_x, n_clust_x, freq_clust_x
  #which currently still hold Nas (bad for math)
  nm <- colnames(df)
  size <- grep('size_clust', colnames(df_eval), fixed=TRUE)
  n <- grep('n_clust', colnames(df_eval), fixed=TRUE)
  freq <- grep('freq_clust', colnames(df_eval), fixed=TRUE)
  nm <- nm[c(size, n, freq)]
  
  
  for(i in 1:length(nm)){
    
    
    df[, nm[i]] <- ifelse(is.na(df[,nm[i]]), 0, df[,nm[i]])
  }
  return(df)
}


only_others <- function(simdat, labels){
  
  grplist <- lapply(strsplit(simdat[simdat$period ==1,]$grp_cmpstn, split = ' ', fixed=TRUE), function(x){as.numeric(x)})
  names(grplist) <- unique(simdat$uid)
  grp_others  <- t(mapply(FUN=function(typ, grp){
    i <- match(typ, grp)
    grp <- grp[-i]
    return(grp)
    
    
  }, grp = grplist, typ= labels))
  colnames(grp_others) <- sprintf('other_t%d', seq(1,dim(grp_others)[2]))
  #grp_others <- split(grp_others, seq(nrow(grp_others)))
  #names(grp_others) <- unique(simdat$uid)
  
  return(grp_others)
  
  
}

normalize <- function(x){
  
  return ((x - min(x)) / (max(x) - min(x)))
}


internal_picker <- function(comparison, cvi=c('Sil', 'D', 'COP', 'DB', 'DBstar', 'CH', 'SF'), algos = NA){
  
  erro <- c('COP', 'DB', 'DBstar')
  cvi_frame <- data.frame()
  if(!is.null(comparison[["pick"]])){
    print('returned unchanged object')
    return(comparison)
  }else{
    
    if(is.na(algos)){
      algos <- c()
      algos <- names(comparison[['results']])
      algos <- unique(algos)
    }
    if(is.null(algos) | length(algos) < 1 ){
      print('failed to extract algorithms, something is wrong with the object')
      comparison[['pick']] <- NULL
      return(comparison)
    }
    for(algo in algos){
      if(class(try(comparison[['results']][[algo]])) == 'try-error'){
        print('something is up with the algorithm...level does not exist')
        print('aborting here an writing try-error')
        comparison[['pick']] <- NULL
        return(comparison)
      }
      
      
      mask <- cvi %in% colnames(comparison[['results']][[algo]])
      use_cols <- c('config_id', cvi[mask])
      
      #invert errounous inversion
      subinv_list <- erro[erro %in% cvi[mask]]
      
      comparison[['results']][[algo]][subinv_list] <- 1/comparison[['results']][[algo]][subinv_list]
      
      if(dim(cvi_frame)[1] == 0){
        cvi_frame <- comparison[['results']][[algo]][use_cols]
        cvi_frame$algo <- algo
      }else{
        tmp <- comparison[['results']][[algo]][use_cols]
        tmp$algo <- algo
        cvi_frame <- rbind(cvi_frame, tmp)
      }
    }
    
    #set NAs to 0
    
    cvi_frame[is.na(cvi_frame)] <- 0
    
    #invert the ones to be minimized
    for(c in erro){
      cvi_frame[,c] <- -cvi_frame[,c]
    }
    
    #normalize the indices to 0-1
    for(col in colnames(cvi_frame)[!(colnames(cvi_frame) %in% c('config_id', 'algo'))]){
      
      cvi_frame[,col] <- normalize(cvi_frame[,col])
    }
    
    
    # cvi_frame$Sil <- normalize(cvi_frame$Sil)
    # cvi_frame$D <- normalize(cvi_frame$D)
    # cvi_frame$COP <- normalize(cvi_frame$COP)
    # cvi_frame$DB <- normalize(cvi_frame$DB)
    # cvi_frame$DBstar <- normalize(cvi_frame$DBstar)
    # cvi_frame$CH <- normalize(cvi_frame$CH)
    # cvi_frame$SF <- normalize(cvi_frame$SF)
    
    ##make median of cvis
    med <- apply(cvi_frame[, use_cols[-1]], MARGIN = 1, function(x){
      median(t(x))
    }
    )
    cvi_frame$median <- med
    #pick via median
    candidates <- cvi_frame[cvi_frame$median == max(cvi_frame$median),]
    if(dim(candidates)[1] > 1){
      #if median not singular, pick via maximum
      sm <- apply(candidates[, use_cols[-1]], MARGIN = 1, function(x){
        sum(t(x))
      }
      )
      candidates$sum <- sm
      candidates <- candidates[candidates$sum == max(candidates$sum),]
      if(dim(candidates)[1] > 1){
        #if also not singular, pick a random one
        candidates <- candidates[sample(1, dim(candidates)[1], 1),]
      }
    }
    
    
    #take the selected config and write into appropriate place
    #for some reason we have the same config twice (is a partitional problem)
    #oncce it is the config and the second time it is the config executed
    candidates <- candidates[1,]
    algo <- candidates$algo
    config_id <- candidates$config_id
    config <- comparison[['results']][[algo]][comparison[['results']][[algo]]$config_id == config_id, ]
    
    
    comparison[["pick"]] <- list()
    comparison[["pick"]][["config"]] <- config
    comparison[["pick"]][["object"]] <- comparison[[paste('objects', candidates$algo, sep='.')]][[candidates$config_id]]
    print('made the custom pick...')
    return(comparison)
  }
  
}

internal_ranking <- function(comparison, cvi=c('Sil', 'D', 'COP', 'DB', 'DBstar', 'CH', 'SF'), algos = NA,
                             add_rank = TRUE, add_normalized = TRUE, add_inverted = TRUE, config_object_only = FALSE){
  set.seed(12345)
  
  erro <- c('COP', 'DB', 'DBstar')
  
  cvi_frame <- data.frame()
  
  if(is.na(algos)){
    algos <- c()
    algos <- names(comparison[['results']])
    algos <- unique(algos)
  }
  
  if(is.null(algos) | length(algos) < 1 ){
    print('failed to extract algorithms, something is wrong with the object')
    comparison[['pick']] <- NULL
    return(comparison)
  }
  for(algo in algos){
    if(class(try(comparison[['results']][[algo]])) == 'try-error'){
      print('something is up with the algorithm...level does not exist')
      print('aborting here and writing try-error')
      comparison[['pick']] <- NULL
      return(comparison)
    }
    
    
    mask <- cvi %in% colnames(comparison[['results']][[algo]])
    use_cols <- c('config_id', cvi[mask])
    
    
    #invert errounous inversion
    subinv_list <- erro[erro %in% cvi[mask]]
    
    comparison[['results']][[algo]][subinv_list] <- 1/comparison[['results']][[algo]][subinv_list]
    
    if(dim(cvi_frame)[1] == 0){
      cvi_frame <- comparison[['results']][[algo]][use_cols]
      cvi_frame$algo <- algo
    }else{
      tmp <- comparison[['results']][[algo]][use_cols]
      tmp$algo <- algo
      cvi_frame <- rbind(cvi_frame, tmp)
    }
  }
  
  
  #set NAs to 0
  
  cvi_frame[is.na(cvi_frame)] <- 0
  
  #invert the ones to be minimized
  
  for(c in subinv_list){
    #print(colnames(cvi_frame))
    #print(subinv_list)
    cvi_frame[,c] <- -cvi_frame[,c]
  }
  
  if(add_inverted){
    mask <- colnames(cvi_frame) %in% c(erro, 'config_id')
    cvi_inv <- cvi_frame[, mask]
    mask <- colnames(cvi_inv) %in% erro
    names(cvi_inv)[mask] <- paste(colnames(cvi_inv[,mask]), 'inverted', sep='_')
    
    
    comparison[['results']] <- lapply(comparison[['results']], function(ls, ranks){
      merge(ls, ranks, by.x = 'config_id', by.y = 'config_id', all.x = TRUE, all.y = FALSE)
    }, ranks= cvi_inv)
    
    
    
  }
  
  #normalize the indices to 0-1
  for(col in colnames(cvi_frame)[!(colnames(cvi_frame) %in% c('config_id', 'algo'))]){
    
    cvi_frame[,col] <- normalize(cvi_frame[,col])
  }
  
  
  # cvi_frame$Sil <- normalize(cvi_frame$Sil)
  # cvi_frame$D <- normalize(cvi_frame$D)
  # cvi_frame$COP <- normalize(cvi_frame$COP)
  # cvi_frame$DB <- normalize(cvi_frame$DB)
  # cvi_frame$DBstar <- normalize(cvi_frame$DBstar)
  # cvi_frame$CH <- normalize(cvi_frame$CH)
  # cvi_frame$SF <- normalize(cvi_frame$SF)
  
  
  if(length(colnames(cvi_frame)[colnames(cvi_frame) %in% cvi]) >1){
    
    cvi_frame[, colnames(cvi_frame) %in% cvi] <- apply(cvi_frame[, colnames(cvi_frame) %in% cvi],
                                                       MARGIN = 2,
                                                       FUN= round, digits = 3)
    dimnames(cvi_frame)[[1]] <- cvi_frame$config_id
    ranks <- as.data.frame(apply(cvi_frame[, colnames(cvi_frame)[colnames(cvi_frame) %in% cvi]],
                                 MARGIN = 2, FUN = rank, ties.method = 'max')
    )
  }else{
    
    cvi_frame[, colnames(cvi_frame) %in% cvi] <- round(cvi_frame[, colnames(cvi_frame) %in% cvi], digits=3)
    ranks <- data.frame(placeholder = rank(cvi_frame[, colnames(cvi_frame) %in% cvi],
                                           ties.method = 'max'))
    
    colnames(ranks) <- colnames(cvi_frame)[colnames(cvi_frame) %in% cvi]
    row.names(ranks) <- cvi_frame$config_id
    row.names(cvi_frame) <- cvi_frame$config_id
    
  }
  
  
  colnames(ranks) <- paste('rank', colnames(ranks), sep='_')
  ranks$rank_sum <- apply(ranks, MARGIN = 1, FUN = sum)
  cvi_frame <- merge(cvi_frame, ranks, by='row.names')
  ind <- which(cvi_frame$rank_sum == max(cvi_frame$rank_sum))
  if(length(ind) >1){
    
    ind <- sample(ind, 1)
    #ind <- ind[1]
  }
  
  if(add_rank){
    print('add ranking to the results...')
    
    comparison[['results']] <- lapply(comparison[['results']], function(ls, ranks){
      merge(ls, ranks, by.x = 'config_id', by.y = 'row.names', all.x = TRUE, all.y = FALSE)
    }, ranks= ranks)
    
    if(add_normalized){
      mask <- colnames(cvi_frame) %in% c(cvi, 'config_id')
      cvi_norm <- cvi_frame[, mask]
      mask <- colnames(cvi_norm) %in% cvi
      names(cvi_norm)[mask] <- paste(colnames(cvi_norm[,mask]), 'normed', sep='_')
      comparison[['results']] <- lapply(comparison[['results']], function(ls, ranks){
        merge(ls, ranks, by.x = 'config_id', by.y = 'config_id', all.x = TRUE, all.y = FALSE)
      }, ranks= cvi_norm)
      
      
    }
  }
  
  
  
  candidates <- cvi_frame[ind[1],]
  algo <- candidates$algo
  config_id <- candidates$config_id
  config <- comparison[['results']][[algo]][comparison[['results']][[algo]]$config_id == config_id, ]
  
  
  comparison[["pick"]] <- list()
  comparison[["pick"]][["config"]] <- config
  comparison[["pick"]][["object"]] <- comparison[[paste('objects', candidates$algo, sep='.')]][[candidates$config_id]]
  
  print('made custom pick...')
  
  if(config_object_only){
    return(list(comparison[["pick"]][["config"]], comparison[["pick"]][["object"]]))
  }else{
    return(comparison) 
  }
}

partial_grid <- function(grid, vec){
  
  #numer of rows
  if(!is.null(dim(grid))){
    rw <- dim(grid)[1]
  }else{
    rw <- length(grid)
  }
  
  partsize <- floor(rw/vec[1])
  if(partsize == 0){
    partsize <- ceiling(rw/vec[1])
  }
  
  rowids <- seq(1, vec[1], 1)
  for(i in 1:vec[1]){
    rowids[i] <- partsize*i
    
  }
  #if(rowids[vec[1]] > rw){
  #  rowids[vec[1]] <- rw
  #}
  if(rowids[vec[1]] < rw){
    diff <- abs(rowids[vec[1]] - rw)
    for(i in 1:(vec[1]-1)){
      tmp <- diff -i 
      if(tmp == 0){
        break
      }
      rowids[vec[1]-i] <- rowids[vec[1]-i]+tmp
    }
    rowids[vec[1]] <- rw
  }
  if(vec[2] != 1){
    if(rowids[vec[2]] > rw & rowids[vec[2]-1] >= rw){
      print("nothing left to do for me...quit")
      quit(save = "no", status = 0, runLast = FALSE) 
    }else if(rowids[vec[2]] > rw & rowids[vec[2]-1] < rw){
      rowids[vec[2]] <- rw
    }
  }else if(vec[2] == 1){
    if(rowids[vec[2]] > rw & rowids[vec[2]+1] >= rw){
      rowids[vec[2]] <- rw
    }
  }
  #failsafe
  if(rowids[vec[1]] != rw){
    rowids[vec[1]] <- rw
  }
  ret <- c(NA, NA)
  if(vec[2] != 1){
    ret[1] <- rowids[vec[2]-1]+1
    ret[2] <- rowids[vec[2]]
  }else{
    ret[1] <- 1
    ret[2] <- rowids[1]
    
  }
  if(partsize == 0 & vec[2] <= rw){
    ret[1] <- vec[2]
    ret[2] <- vec[2]
  }else if(partsize == 0 & vec[2] > rw){
    print("nothing left to do for me...quit")
    quit(save = "no", status = 0, runLast = FALSE)    
  }
  
  return(ret)
}


get_results <- function(comparison_res, names, gridinfo, savepath){
  require(dplyr)
  dat <- NULL
  for(i in 1:length(names)){
    
    if(is.null(dat)){
      dat <- comparison_res[[names[i]]]
      dat$algo <- names[i]
    }else{
      comparison_res[[names[i]]]$algo <- names[i]
      dat <- dplyr::bind_rows(dat, comparison_res[[names[i]]])
    }
  }
  
  for(coln in colnames(gridinfo)){
    if(coln !='idt'){
      dat[, paste('var', coln, sep='_')] <- as.numeric(as.character(gridinfo[,coln]))
    }else{
      dat[, paste('var', coln, sep='_')] <- as.character(gridinfo[,coln])
    }
  }
  
  print('saving result data...')
  saveRDS(dat, file=savepath)
  
}

get_fun <- function(fun){
  fun <- deparse(fun)
  chunk <- tail(fun, 1)
  words <- strsplit(chunk, "\"")[[1]]
  return(words[2])
}


norm_cvis <- function(df, cvicols = c("Sil","D", "COP","DB","DBstar","CH" , "SF"),
                      erro = c('COP', 'DB', 'DBstar')){
  
  set.seed(12345)
  #invert errounous inversion
  subinv_list <- erro[erro %in% cvicols]
  df[cvicols][subinv_list] <- 1/df[cvicols][subinv_list]
  
  #make them negative to invert them 
  df[cvicols][subinv_list] <- -1*(df[cvicols][subinv_list])
  
  #get min and max for norming
  
  mins <- apply(df[,cvicols], 2, min)
  maxs <- apply(df[,cvicols], 2, max)
  normed <- as.data.frame(t(apply(apply(df[, cvicols], 1, '-', mins), 2, '/', maxs-mins)))
  colnames(normed) <- paste(colnames(normed), 'normed', sep='_')
  
  df <- dplyr::bind_cols(df, normed)
  
  #check if NAs in normed frame (can happen when subgroups have only one value)

  df[,colnames(normed)] <- as.data.frame(apply(as.matrix(colnames(normed)), 1, function(nm, dat){
    ind = which(is.na(dat[,nm]))
    if(length(ind) ==0){
      return(dat[,nm])
    }
    nmo <- strsplit(nm, '_', fixed=TRUE)[[1]][1]
    dat[ind,nm] <- ifelse(dat[ind, nmo] >0, 1, 0) 
    return(dat[,nm])
    
  }, dat=df))
  
  return(df)
  
}

rank_cvi<- function(df, cvicols=c("Sil_normed","D_normed","COP_normed","DB_normed",
                                  "DBstar_normed","CH_normed","SF_normed")){
  
  df[, cvicols] <- apply(df[, cvicols], MARGIN = 2, FUN= round, digits = 3)
  ranks <- as.data.frame(apply(df[, cvicols], MARGIN = 2, FUN = rank, ties.method = 'max'))
  colnames(ranks) <- paste(colnames(ranks), 'rank', sep='_')
  
  
  df <- dplyr::bind_cols(df, ranks)
  
  return(df)
  
}


select_highest <- function(df, ranknames = c("Sil_normed_rank","D_normed_rank","COP_normed_rank","DB_normed_rank",
                                             "DBstar_normed_rank","CH_normed_rank","SF_normed_rank"), select = FALSE){
  
  set.seed(123456)
  
  df$rank_sum <- apply(df[,ranknames], MARGIN = 1, FUN = sum)
  ind <- which(df$rank_sum == max(df$rank_sum))
  
  if(length(ind) >1 & select){
    ind <- sample(ind, 1)
  }
  
  return(df[ind,])
  
}


rank_results <- function(df, cvicols= c("Sil","D", "COP","DB","DBstar","CH" , "SF")){
  df <- norm_cvis(df, cvicols)
  cvicols <- paste(cvicols, 'normed', sep='_')
  df <- rank_cvi(df, cvicols)
  
  
  return(df)
  
}

select_ranked_result <- function(df, cvicols= c("Sil","D", "COP","DB","DBstar","CH" , "SF"), select =FALSE){
  
  df <- rank_results(df, cvicols)
  #cvicols <- paste(cvicols, 'rank', sep='_')
  highest <- select_highest(df, cvicols, select)
  
  return(highest)
  
  
}

calc_cvi_point <- function(df, rank, FUN= median, cvicols= c("Sil","D", "COP","DB","DBstar","CH" , "SF"), select =FALSE){
  
  if(rank){
    df <- select_ranked_result(df, cvicols, select)
  }else{
    df<- norm_cvis(df, cvicols)
  }
  
  
  cvicols <- paste(cvicols, 'normed', sep='_')
  
  df <- as.data.frame(do.call(cbind, lapply(df[,cvicols], FUN)))
  
  colnames(df) <- paste(cvicols, get_fun(FUN), sep='_')
  
  return(df)
  
}


calc_along <- function(df, along, rank, FUN=median, cvicols= c("Sil","D", "COP","DB","DBstar","CH" , "SF")){
  require(dplyr)
  df <- df %>% group_by_at(along) %>% group_modify(~calc_cvi_point(df=.x, rank=rank, FUN=FUN, cvicols = cvicols))
  return(df)
  
}


find_cutoff_k <- function(df, cutofflow, cutoffup, valuecol='value'){
  
  sorted <- df[order(df[,valuecol], decreasing = TRUE),]
  low = dim(sorted)[1]
  up = 0
  for(i in 1:dim(sorted)[1]){
    if(sorted[i,valuecol] > cutofflow){
      next
      }
    if(low > sorted[i,'k']){
      low <- unname(unlist(sorted[i,'k']))
      }
    if(sorted[i,valuecol] <= cutoffup){
      up <- unname(unlist(sorted[i-1,'k']))
      break
      }
    }
  return(list(low, up))
}




