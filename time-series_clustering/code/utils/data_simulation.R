#@author = Marcel Schubert
##Data Simulation Script to Simulate Group Interactions in a Public-Goods Game
####Change parameters here as needed

#remainder-error rnorm(N, 1, eta)
#eta <- 0.3

#individual-specific error rnorm(N,1,sigma)
#sigma <- 0.3

#type vector
# type 1: 0 + err; #freerider
# type 2: conditional up till 10, reverse conditional thereafter; ##hump-shaped
# type 3: conditional + err; 
# type 4: unconditionally high;
# type 5: far-sighted free-rider: behaves like conditional cooperator until period p, and free-rides thereafter
#if more types are added one has to adapt gen_strategy_tables() and gen_time_dep_strategy()

#types <- c(1,2,3,4,5)

#groupsize
#grpsize <- 4

#number of rounds to play
#numrounds <- 10

#endowment
#endowment <-20

#random contribution in first round or contribution as laid out in the strategy functions
#rand_first <- FALSE

#draw random subset from all possible group compositions; values are TRUE, FALSE or string 'specific'
#if 'specific' the arguement group_compositions must be given as a list and number of players in group must equal those in the composition vector
#if random_subset == FALSE, then n is the times each possible group compositions is in the data set
#random_subset <- FALSE

#if specific specify compositions here, otherwise arguement is ignored and does not need to be passed to function
#group_compositions <- list(c(4,4,4,5))#, c(1,2,3,4), c(5,5,5,5))

#otherwise n is the number of groups in the random subset; it must be n>=2
#n <- 5

#do if want to add errors
#add_errors <- FALSE

#individual specific error
#sigma <- 0.3

#remainder error
#eta <- 0.3

#seed - every time seed is used it is a function depending on this value; value itself is not reused
#seed <- 12345


##### flexible data generation solution


gen_all_combinations <- function(types = c(1,2,3,4,5), grpsize){
  #generate all possible combinations, return as list
  require(iterpc)
  
  I = iterpc(n=length(types), r=grpsize, labels = types, ordered = FALSE, replace = TRUE)
  res = as.list(data.frame(t((getall(I)))))
  rm(I)
  
  return(res)
}

gen_strategy_tables <- function(endowment, grpsize){
  ##if additional types are introduced, one has to put the contribution rule here
  #position in list corresponds to numerical identifier of type
  
  
  contrib_pos <- 0:endowment
  #avgothercontrib = (0:(endowment*(grpsize-1)))/(grpsize-1)
  rules = list()
  #_f marks as timeindependent strategy, _t marks as timedependent
  nomen <- c('freerider_f', 'hump-shaped_f', 'conditional_f', 'unconditional_f', 'farsighted-freerider_t')
  ##in first place of table there is the initial contribution
  #freerider
  rules[[1]] <- rep(0, length(contrib_pos)+1)
  
  #hump-shaped
  rules[[2]] <- c(round(endowment/4),ifelse(contrib_pos <= round(endowment/2), contrib_pos, endowment-contrib_pos))
  
  #conditional
  rules[[3]] <- c(round(endowment/2), contrib_pos)
  
  #unconditional
  
  rules[[4]] <- rep(endowment, length(0:endowment)+1)
  
 
  #far-sighted freerider
  
  rules[[5]] <- c(round(endowment/2), rep(0, length(1:endowment)))
  
  names(rules) <- nomen
  
  return(rules)
}

gen_time_dep_strategy <- function(endowment, grpsize, numrounds){
  ##if additional time-dependent types are introduced, one has to put the contribution rule here
  nomen <- c('farsighted-freerider_t')
  
  #avgothercontrib = (1:(endowment*(grpsize-1)))/(grpsize-1)
  rules_t <- list()
  #far-sighted freerider, this denotes the periods and as which type the the player plays in this period
  rules_t[[1]] <- c(rep(3, numrounds-(floor(numrounds/2))), rep(1, floor(numrounds/2)))
  
  names(rules_t) <- nomen
  
  return(rules_t)
  
}

gen_contribution_rules <- function(endowment, grpsize, numrounds){
  
  rules = list()
  nomen <- c('time_independent', 'time_dependent')
  rules[[1]] <- gen_strategy_tables(endowment, grpsize)
  
  rules[[2]] <- gen_time_dep_strategy(endowment, grpsize, numrounds)
  names(rules) <- nomen
  
  return(rules)
}

draw_groups <- function(n, combinations, random_subset = FALSE, seed = 12345){
  set.seed(seed)
  if(random_subset == TRUE){
    sub <- combinations[sample(1:length(combinations), size = n, replace = TRUE)]
    
  }
  
  else{
    sub <- combinations[rep(c(1:length(combinations)), n)]
    
  }
   return(sub)
  
}

chunk2 <- function(x,n){
  ##vector into list of equal sized chunks
  split(x, cut(seq_along(x), n, labels = FALSE))
}
  
make_errors <- function(sigma, eta, numgroups, grpsize, rounds, add_errors = TRUE, seed=12345){
  
  set.seed(seed*2)
  ##individualspecific errors
  errors <- chunk2(rnorm(numgroups*grpsize, 0, sigma), numgroups)
  
  ##roundspecific errors/residual error
  errors_round <- lapply(chunk2(rnorm(numgroups*grpsize*rounds, 0, eta), numgroups), chunk2, n=rounds)
  ##add remainder
  ##do for every list of errors by group
   errors <- as.list(data.frame(mapply(function(eind, eround){
    ##here we have the individual specific errors in vector with #elements == #members of group
    ##the remainder are in a list with #elements == #rounds and each element is vector with #elements of vector == #members of group
            lapply(eround, function(eroun, ein){
                              eroun + ein
                            }, ein= eind)
                    }, eind=errors, eround=errors_round)))

   if(!add_errors){
     errors <- lapply(errors, function(x){lapply(x, function(y){y <- rep(0, length(y))})})
     
   }
   
   return(errors)
}

create_id <- function(marker, gid){
  
  uid <- as.numeric(paste(gid, marker, sep=''))
  return(uid)
}

calc_contrib_others <- function(uid, contribution){
  ##this calculates the contrib of all others in group when a single group for a single period is given
  
  avg_others <- sapply(uid, function(x, id, contr){
    avg <- sum(contr[!(x == id)])/(length(contr)-1)
  }, id =uid, contr = contribution)
  

  return(round(avg_others))
}

make_group_wise_contributions <- function(group, errors, group_id, endowment, numrounds, strategy_f, strategy_t, rand_first){

  #function makes the contributions on individual group-level and returns them
  
  #uid
  uid <- sapply(1:length(group), create_id, gid=group_id)
  
  #make first round contribution
  
  #if random first round contrib
  if(rand_first){
    contr <- sample(0:20, 4)+errors[[1]]
    contr <- ifelse(contr <= endowment, ifelse(contr >= 0, contr, 0), endowment)
    }
  #if fixed first round contrib from strategy table
  else{
  contr <- mapply(function(individ, error){
    contr <- round(strategy_f[[individ]][1]+error)
    #check that contribution is within limits
    contr <- ifelse(contr <= endowment, ifelse(contr >= 0, contr, 0), endowment)
  }, individ = group, error = errors[[1]])
  }
  contr <- round(contr)
  #make vector with croup composition so that every member of group has the same entry
  grp_cmpstn <- rep(paste(group, collapse = ' '),length(group))

  df <- data.frame(uid = uid, group_id = rep(group_id, length(group)), contribution =contr, grp_cmpstn = grp_cmpstn,
                   avg_others_l = rep(-1, length(group)), type = group, period = rep(1, length(group)))

  ##calculate contributions of other rounds
  for(i in 2:numrounds){
    #for each individual in group
     for(ind in uid){
     
       #position in the uid vector is the same as the position of individual in the group/type vector
       #cmd <- sprintf('period %s', i)
       #print(eval(cmd))
       
       pos = which(ind == uid)[[1]]

       tp <- group[pos]
       avg_others <- round(sum(df[df$period == i-1 & df$uid != ind,]$contribution)/(length(group)-1))

       
       #get name of type
       nom <- names(strategy_f[tp])[1]

       #retrieve contribution from strategy table
       ##if time dependent
       
       if(grepl('_t', nom, fixed = TRUE)){
         contr <- strategy_f[[strategy_t[[nom]][i]]][avg_others+2]
         #print(contr)
         
         }
       #if fixed strategy
       else{
         
         contr <- strategy_f[[tp]][avg_others+2]
         
       }
       
       ##add random error to contribution
       
       contr <- round(contr + errors[[i]][pos])

       contr <- ifelse(contr <= endowment, ifelse(contr >= 0, contr, 0), endowment)
       
       tmp <- data.frame(uid = ind, group_id = group_id, contribution = contr, grp_cmpstn = grp_cmpstn[1],
                         avg_others_l = avg_others, type = tp, period = i)
       
       df <- rbind(df, tmp)
       
       #if(i >3){return(df)}
       
     }
    
  } 
  
  
  return(df)
}

make_contributions <- function(endowment, errors, groups, numrounds, strategy_f,
                               strategy_t, rand_first = FALSE, seed = 12345){
  ##function to make the contributions and concat the datarame correctly
  
  require(dplyr)
  set.seed(floor(seed/3))
  contrib_matrix <- mapply(make_group_wise_contributions, group=groups, errors=errors, group_id = 1:length(groups),
         MoreArgs = c(endowment=endowment, numrounds = numrounds, strategy_f = list(strategy_f), strategy_t = list(strategy_t),
                      rand_first = rand_first))
  
  #rearrange as data frame from matrix-like
  ##make df with first entry then loop; row and cols are named but col-names are not necessarily unique -> index access
  df <- data.frame(uid = contrib_matrix['uid', 1][[1]], group_id = contrib_matrix['group_id', 1][[1]],
                   contribution= contrib_matrix['contribution', 1][[1]], grp_cmpstn=contrib_matrix['grp_cmpstn', 1][[1]],
                   avg_others_l=contrib_matrix['avg_others_l', 1][[1]], type = contrib_matrix['type', 1][[1]],
                   period=contrib_matrix['period', 1][[1]])

  for(i in 2:(dim(contrib_matrix)[2])){
    
    tmp <- data.frame(uid = contrib_matrix['uid', i][[1]],
                      group_id = contrib_matrix['group_id', i][[1]],
                      contribution= contrib_matrix['contribution', i][[1]],
                      grp_cmpstn=contrib_matrix['grp_cmpstn', i][[1]],
                      avg_others_l=contrib_matrix['avg_others_l', i][[1]],
                      type = contrib_matrix['type', i][[1]],
                      period=contrib_matrix['period', i][[1]])
    
    
    df <- rbind(df, tmp)
  }
  

  rm(contrib_matrix)
  
  
  #df %>% group_by(group_id, period) %>%
  #          mutate(avg_others = calc_contrib_others(uid, contribution))
  
  #df %>% group_by(group_id, uid) %>%
  #        mutate(avg_others_l = c(-1, avg_others[1:(length(avg_others)-1)]))
  return(df)
  
}

generate_data <- function(types, grpsize, numrounds, endowment, rand_first, n, random_subset, 
                          add_errors, sigma = 0.3, eta = 0.3, seed = 12345, group_compositions = NA, normalize = TRUE){
  #function generates all data; call only this function
  
  
  comb_lookup <- group_compositions
  #generate lookup-list with all combinations if not 'specific' otherwise do not execute
  if(paste(random_subset) != 'specific'){
    print('generating all possible group compositions...')

    comb_lookup <- gen_all_combinations(types, grpsize)
  }
  
  stopifnot(!is.na(comb_lookup))
  
  #generate all strategies [[1]] are the fixed ones [[2]] are the time-dependent ones
  strategies <- gen_contribution_rules(endowment, grpsize, numrounds)

  #generate groups that played from lookup table
  print('draw groups in data...')
  groups <- draw_groups(n, comb_lookup, random_subset, seed)
  rm(comb_lookup)
  
  #generate errors; if add_errors == FALSE, the a list full of zeros and nothing will be added
  print('generate necessary errors...')
  errors <- make_errors(sigma, eta, length(groups), grpsize, numrounds, add_errors, seed)
  

  #generate data
  print('generate data...')
  df <- make_contributions(endowment,errors, groups, numrounds, strategies[['time_independent']],
                           strategies[['time_dependent']], rand_first, seed)
  print('finished generating')
  
  
  ################
  # add measures #
  ################
  

  #ratio
  df$ratio<-(df$contribution+1)/((df$avg_others_l+1))
  df$ratio[!is.finite(df$ratio)] <- max(df$ratio[is.finite(df$ratio)])
  

  #difference
  df$diff<- df$contribution - df$avg_others_l
  
  #check if normalization should happen
  if(normalize){
    df$contribution <- (df$contribution - 0)/(endowment - 0)
    df$avg_others_l <- (df$avg_others_l - 0)/(endowment - 0)
    
    sb <- df$ratio[df$period != 1]
    df$ratio <- (df$ratio - min(sb))/(max(sb)-min(sb))
    df$ratio[is.na(df$ratio)] <- 0
    ##slightly above zero to avoid complications
    #df$ratio <- df$ratio + 0.0001
    
    df$diff <- (df$diff - (-endowment))/(endowment - (-endowment))
      
    }
  
  
  
  return(df)
}


#group_compositions <- list(c(1,2,3,4))

#df <- generate_data(types,grpsize, numrounds, endowment, rand_first, n, random_subset='specific', 
#              add_errors, sigma, eta, seed, group_compositions)

