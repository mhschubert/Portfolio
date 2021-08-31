#@author = Marcel Schubert

write_metrics <-function(df, clusresult, typeresult, matching, df_colnames, types, clustern){

  #function writes metrics into eval dataframe
  metrics <- c("accuracy", "precision", "recall", "f1")
  
  
  ##if not whole upper bound for clusters or all possible types are present
  numclust <- unique(clusresult$clus)
  numTyp <- unique(typeresult$type)
  
  clustern <- numclust


  types <- numTyp
  

  
  
  #make for cluster level


  for(i in 1:length(metrics)){
  

    avg = 0
    for(j in 1:length(clustern)){
 
  
      
      
      metric <- clusresult[clusresult$clus == clustern[j], paste(metrics[i], '_clus', sep='')][1]
      if(is.null(metric)){
        metric <-0
      }
      if(is.na(metric)){
        metric <- 0
      }
      
    
      df[1, sprintf(paste(metrics[i], '_cluster_%d', sep=''), j)] <- metric
      
      
      avg = avg + metric
      
    }
    
    avg = avg/length(clustern)
 
    df[1, paste('avg_', metrics[i], '_cluster', sep='')] <- avg
  }
  
  #make for type level


  for(i in 1:length(metrics)){
  
    avg = 0
    for(j in 1:length(types)){
     
     
      metric <- typeresult[typeresult$type == types[j], paste(metrics[i], '_typ', sep='')][1] ##important, type may appear more than once in df but all metric values are the same
      if(is.null(metric)){
        metric <-0
      }
      if(is.na(metric)){
        metric <- 0
      }
      
      
      df[1, sprintf(paste(metrics[i], '_type_%d', sep=''), j)] <- metric
      
      avg = avg + metric
      
    }
    
    avg = avg/length(types)

    df[1, paste('avg_', metrics[i], '_type', sep='')] <- avg
  }
  
  #add entries for dic mapping


  match <- as.list(matching)
  clus <- names(match)
  for(i in 1:length(match)){
    
    df[1, sprintf('pred_clust_type_%s', paste(clus[i]))] <- match[[clus[i]]]
  }
  
  return(df)
  
}

accuracy <- function(true_pos, true_neg, false_pos, false_neg, ...){
  
  acc <- (true_neg+true_pos)/(true_pos+true_neg+false_pos+false_neg)

  return(acc)
  
}

precision <- function(true_pos, true_neg, false_pos, false_neg, ...){
  if(true_pos+false_pos==0){
    print('Warning: Division by Zero, precision is ill-defined; Setting it to 0'
    )
    prec <- 0
  }
  else{
    prec <- true_pos/(true_pos+false_pos)
  }
  return(prec)
  
}

recall <- function(true_pos, true_neg, false_pos, false_neg, ...){
  if(true_pos+false_pos==0){
    print('Warning: Division by Zero, recall is ill-defined; Setting it to 0'
    )
    rec <- 0
  }
  else{
    rec <- true_pos/(true_pos+false_neg)
  }
  return(rec)
}

f1 <- function(true_pos, true_neg, false_pos, false_neg, ...){
  
  dots <- list(...)

  if('recall_weight' %in% names(dots)){

    recall_weight <- dots[['recall_weight']]
    
  }
  else{
    
    recall_weight <- 0.5
  }
  if(true_pos+false_pos == 0){
    print('Warning: F1-score is ill-defined if no positives are predicted for class')
  }
  prec <- precision(true_pos, true_neg, false_pos, false_neg)
  rec <- recall(true_pos, true_neg, false_pos, false_neg)
  if(prec+rec == 0){
    print('Setting f1-score to 0 as well')
    f1 <- 0
  }
  else{
    f1 <- (1+recall_weight**2)*(prec*rec)/((prec*recall_weight**2)+rec)
  }
}

interface_eval_functions <- function(true_types, predicted_types, FUN, true_label, multiclass = FALSE, ...){

  #if not multiclass problem
  if(!multiclass){
    
      true_pos <- sum((true_types == true_label) & (predicted_types == true_label))
      true_neg <- sum((true_types != true_label) & (predicted_types != true_label))
      false_pos <- sum((true_types != true_label) & (predicted_types == true_label))
      false_neg <- sum((true_types == true_label) & (predicted_types != true_label))
      
      res <- FUN(true_pos, true_neg, false_pos, false_neg, ...)

      
      
      return(res)
  }
  else{
    print('NOT IMPLEMENTED YET')
  }
  
}

make_df <- function(comparison, labels = NA, precomputed=FALSE, tsobject = FALSE){
  res <- list()
  
  require(hash)
  require(dplyr)
  
  if(!precomputed){
    ##check if labels object is given otherwise check for object in environment
    if(class(labels) != "logical"){
      type <- labels
  

    }else{
      type <- simdat_grp$type[simdat_grp$period==2]
    }
    #create df with true type colum and cluster column

    if(!tsobject){
      dl<-comparison[["pick"]][["object"]]@datalist
      clus<-comparison[["pick"]][["object"]]@cluster
    }else{
      dl <- comparison@datalist
      clus <- comparison@cluster
    }
    

    df_clus<-data.frame(clus = clus, type = as.numeric(as.character(type)))
    df_clus$uid<-names(dl)
  }else{
    
    df_clus <- comparison %>% distinct(uid, .keep_all = TRUE)
    df_clus <- df_clus[, c('uid', 'clus', 'type')]
    
    
  }
  
  
  #create df with type colum and cluster column as well as frequency of type within cluster
  df_clus %>%
    group_by(clus,type) %>%
    summarise(n = n(), .groups="keep") %>% group_by(clus, .add =FALSE) %>%
    mutate(freq = n / sum(n)) -> df_clus1
  df_clus <- ungroup(df_clus)
  

  # make df providing a mapping betweent type and cluster
  df_clus1 %>% 
    group_by(clus) %>%
    filter(freq == max(freq)) ->df_clus2  #select the dominant type from each cluster
  
  ratio_c_t<-length(unique(df_clus2$type))/length(unique(df_clus$type)) 
  df_clus1 <- ungroup(df_clus1)
  #create dictionary with mapping to implicitly predicted types
  dic <- hash()
  for(i in 1:dim(df_clus2)[1]){
    dic[[paste(ungroup(df_clus2)[i,'clus'])]] <- as.numeric(paste(ungroup(df_clus2)[i,'type']))
  }
  
  ##append predicted type to df
  df_clus <- ungroup(df_clus %>% group_by(uid) %>%
                       mutate(pred_type = dic[[paste(clus)]])
  )
  

  res[[1]] <- df_clus
  res[[2]] <- df_clus1
  res[[3]] <- df_clus2
  res[[4]] <- dic

  return(res)
  
}

eval_result <- function(comparison, labels = NA, recall_weight = 0.5, precomputed = FALSE, tsobject = FALSE){
  
  res <- make_df(comparison, labels, precomputed, tsobject)
  
  df_clus <- res[[1]]
  dic <- res[[4]]
  
  
 
  #make the evaluation metrics on a individual cluster level (well-defined; correct label is positive class, all others are negative class)
  df_clus <- ungroup(df_clus)
  clus_metr <- df_clus %>% group_by(clus) %>%
              summarise(accuracy_clus = interface_eval_functions(type, pred_type, accuracy, true_label = dic[[paste(unique(clus)[1])]]),
                     precision_clus = interface_eval_functions(type, pred_type, precision, true_label = dic[[paste(unique(clus)[1])]]),
                     recall_clus = interface_eval_functions(type, pred_type, recall, true_label = dic[[paste(unique(clus)[1])]]),
                     f1_clus = interface_eval_functions(type, pred_type, f1, true_label = dic[[paste(unique(clus)[1])]], recall_weight = recall_weight)
                     , .groups= "keep")
  df_clus <- ungroup(df_clus)
  res[[3]] <- merge(res[[3]], clus_metr, by ='clus')
  
 
  
  #make evaluation metrics on type-level by varying which one is considered the positive type (well-defined)
  tmp <- data.frame(type=c(NA), accuracy_typ=c(NA), precision_typ=c(NA),recall_typ = c(NA), f1_typ = c(NA))
  type <- df_clus$type
  pred_type <- df_clus$pred_type
  for(i in 1:length(unique(df_clus$type))){
    
    true_label <- unique(df_clus$type)[i]
    tmp[i,] <- c(type = true_label,
                 accuracy_typ = interface_eval_functions(type, pred_type, accuracy, true_label = true_label),
                 precision_typ = interface_eval_functions(type, pred_type, precision, true_label = true_label),
                 recall_typ = interface_eval_functions(type, pred_type, recall, true_label = true_label),
                 f1_typ = interface_eval_functions(type, pred_type, f1, true_label = true_label, recall_weight = recall_weight)
                )
    
  }
  
  res[[2]] <- merge(res[[2]], tmp, by='type')

  return(res)
  
}
